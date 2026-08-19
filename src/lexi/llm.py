"""Thin LLM wrapper: structured output, retries, and token accounting.

Kept deliberately small, and the ONLY module that knows which provider is in
use. Everything else -- agent, tools, retrieval, evals -- depends on this
interface, so swapping providers is a change to one file. That isolation earned
its keep during the build: the system moved from Gemini to Azure-hosted
DeepSeek-V4-Flash without touching anything downstream.

Provider: Azure AI Foundry, which serves DeepSeek over an OpenAI-compatible
`/models/chat/completions` endpoint, so the OpenAI client works against it once
the `api-version` query parameter is supplied.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
# Gemini tells us exactly how long to wait; honour it instead of guessing.
_RETRY_AFTER = re.compile(r"retry in ([\d.]+)s", re.I)


class ContentFiltered(RuntimeError):
    """The provider's safety filter refused the request.

    Distinct from a transient failure: the same input will be refused again, so
    this must be routed around rather than retried. It fires on legitimate legal
    text -- descriptions of fatal accidents, injuries, counterfeiting -- which is
    an occupational hazard of running judgments through a general-purpose filter.
    """



def is_content_filter(exc: BaseException) -> bool:
    """Was this refused by the provider's safety filter?

    Shared by both call paths -- `LLM.complete()` and the tool-bound client the
    agent invokes directly -- because they receive the same provider error and
    must classify it identically. Matches the response body rather than an
    exception type, since the OpenAI-compatible client surfaces it as a generic
    400.
    """
    if isinstance(exc, ContentFiltered):
        return True
    low = str(exc).lower()
    return "content_filter" in low or "content management policy" in low


class RateLimiter:
    """Process-wide minimum interval between API calls.

    Shared across threads because quota is per-deployment, not per-thread. The
    Azure deployment allows 250 requests/minute, so this is comfortable
    headroom rather than the binding constraint it was on a free tier.
    """

    def __init__(self, rpm: int):
        self._min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def pause(self, seconds: float) -> None:
        """Hard stop for everyone -- used when the API reports a quota breach."""
        with self._lock:
            time.sleep(seconds)
            self._last = time.monotonic()


_limiter = RateLimiter(settings.rpm)


@lru_cache(maxsize=8)
def _build_client(model: str, temperature: float):
    """One client per (model, temperature), reused for the process lifetime.

    Constructing ChatOpenAI per call -- which an earlier version did -- creates a
    fresh httpx connection pool every time. Over a long evaluation the sockets
    accumulate faster than they are reclaimed and calls start failing with
    APIConnectionError, which reads like a flaky provider but is self-inflicted.
    Caching keeps a single pool alive and lets keep-alive do its job.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.require_key(),
        base_url=settings.azure_base_url,
        # Azure requires api-version as a query parameter; the OpenAI client has
        # no first-class slot for it, so it rides along on every request.
        default_query={"api-version": settings.azure_api_version},
        max_retries=0,  # retries are handled in complete(), with our own backoff
        timeout=settings.request_timeout_s,
    )


def throttle() -> None:
    """Block until the process-wide limiter grants a slot.

    The tool-bound agent client cannot route through complete(), so it calls
    this instead. Without it the agent's own turns -- the heaviest requests in
    the system, 60-75K tokens late in a research run -- were the only calls NOT
    counted against the rpm budget: three concurrent eval workers saturated the
    deployment's per-minute quota and a 429 killed the control arm's key query.
    """
    _limiter.acquire()


def backoff_429(msg: str) -> float:
    """Shared 429 response: obey the server's stated wait when present, pause
    the process-wide limiter so ALL workers back off together (the bucket is
    shared, so one worker sleeping while two keep spending clears nothing),
    and return the wait for the caller to sleep."""
    m = _RETRY_AFTER.search(msg)
    wait = min(float(m.group(1)) + 1.0 if m else 25.0, 90.0)
    _limiter.pause(wait)
    return wait


@dataclass
class Usage:
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    throttled: int = 0

    def add(self, i: int, o: int) -> None:
        self.calls += 1
        self.in_tokens += i
        self.out_tokens += o


@dataclass
class LLM:
    """Azure AI Foundry client. `model` is injectable so evals can pin a
    different deployment -- which is how the judge and the second gold annotator
    are kept off the agent's own model."""

    model: str = field(default_factory=lambda: settings.chat_model)
    temperature: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def _client(self):
        return _build_client(self.model, self.temperature)

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Rate-limited call with backoff.

        Failure modes handled distinctly:
          429 -> throttled; obey any stated wait, else back off
          5xx -> transient; short exponential backoff
          else -> two quick retries, then surface it
        """
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        last: Exception | None = None

        for attempt in range(settings.max_retries):
            _limiter.acquire()
            try:
                resp = self._client().invoke(msgs)
                meta = getattr(resp, "usage_metadata", None) or {}
                self.usage.add(meta.get("input_tokens", 0), meta.get("output_tokens", 0))
                return _text_of(resp.content)
            except Exception as e:  # noqa: BLE001 - provider raises a wrapped error type
                last = e
                msg = str(e)
                low = msg.lower()

                # Azure's content filter trips on this corpus: judgments describe
                # fatal collisions, injuries and counterfeiting. Retrying is
                # pointless -- the input will be refused identically -- so raise a
                # typed error the caller can route around rather than burning
                # attempts and then failing the whole query.
                if is_content_filter(e):
                    raise ContentFiltered(msg[:300]) from e

                if "429" in msg or "rate limit" in low:
                    m = _RETRY_AFTER.search(msg)
                    wait = float(m.group(1)) + 1.0 if m else 20.0
                    self.usage.throttled += 1
                    _limiter.pause(min(wait, 90.0))
                elif any(c in msg for c in ("500", "502", "503", "504")) or any(
                    c in low for c in ("timeout", "connection", "temporarily", "reset")
                ):
                    # Includes APIConnectionError, which killed a query in an
                    # earlier run because it matched none of the numeric codes.
                    time.sleep(min(2 ** attempt, 20))
                else:
                    if attempt >= 1:
                        raise
                    time.sleep(2)
        raise RuntimeError(f"LLM call failed after {settings.max_retries} attempts: {last}")

    def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        """Return a validated pydantic object, re-prompting once on failure.

        We ask for raw JSON rather than using tool-call binding because the same
        code path has to work for enrichment, reranking and the eval judges --
        one mechanism, one failure mode.
        """
        instruction = (
            f"{prompt}\n\nRespond with ONLY valid JSON matching this schema. "
            f"No prose, no markdown fence.\n\nSCHEMA:\n"
            f"{json.dumps(schema.model_json_schema(), indent=1)}"
        )
        raw = self.complete(instruction, system)
        try:
            return schema.model_validate(_extract_json(raw))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            repair = (
                f"Your previous response did not validate.\n\nERROR:\n{e}\n\n"
                f"PREVIOUS:\n{raw[:3000]}\n\nReturn corrected JSON only."
            )
            return schema.model_validate(_extract_json(self.complete(repair, system)))


def _text_of(content) -> str:
    """Flatten a LangChain message payload to plain text.

    Some models return `content` as a LIST of parts (answer text plus reasoning
    or signature blobs) rather than a string. Stringifying that list yields
    Python repr garbage, so pull out the text parts explicitly. Kept
    provider-agnostic because both Gemini and DeepSeek do this.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
        return "\n".join(p for p in out if p)
    return str(content)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, fenced or not."""
    text = text.strip()
    if m := _FENCE.search(text):
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in response: {text[:300]}")
        return json.loads(text[start : end + 1])
