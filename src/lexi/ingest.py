"""PDF -> clean text -> structure-aware chunks.

The corpus is Indian Kanoon HTML-to-PDF exports. They share a stable shape:

    <case title> on <date>
    Author: <judge>
    Bench: <judges>
    ... body ...

and every page carries a repeating footer:

    <case title> on <date>
    Indian Kanoon - http://indiankanoon.org/doc/<id>/
    <page number>

That footer must be stripped before chunking or it pollutes every passage and
poisons both embeddings and BM25.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

from .config import settings
from .schemas import Chunk

# --- Patterns ----------------------------------------------------------------

_KANOON_URL = re.compile(r"Indian Kanoon\s*-\s*http://indiankanoon\.org/doc/(\d+)/?")
# Title and date both wrap across lines in the source PDFs, so this is matched
# against a whitespace-collapsed header block, NOT line-anchored.
_HEADER_TITLE = re.compile(r"^(.*?)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})")
_AUTHOR = re.compile(r"Author:\s*([^\n]+)")
_BENCH = re.compile(r"Bench:\s*([^\n]+)")
# Widened after measuring: the narrow form missed 13/56 judgments (tribunals,
# "JUDICATURE AT", benches named on a following line).
_COURT = re.compile(
    r"((?:IN\s+THE\s+)?(?:HON'?BLE\s+)?"
    r"(?:HIGH\s+COURT\s+(?:OF\s+JUDICATURE\s+)?(?:AT|OF|FOR)?[^\n]{0,70}"
    r"|SUPREME\s+COURT\s+OF\s+INDIA"
    r"|MOTOR\s+ACCIDENTS?\s+CLAIMS?\s+TRIBUNAL[^\n]{0,50}"
    r"|COURT\s+OF\s+THE?\s+[^\n]{0,60}))",
    re.I,
)

# Section markers common to Indian judgments, in the order they normally appear.
_SECTION_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("order", re.compile(r"^\s*(ORDER|O R D E R)\s*$", re.M)),
    ("judgment", re.compile(r"^\s*(JUDGMENT|J U D G M E N T|ORAL JUDGMENT)\s*$", re.M)),
]

# "12." or "12)" at the start of a line -- Indian judgments number their paragraphs.
_PARA_NUM = re.compile(r"^\s*(\d{1,3})[\.\)]\s+", re.M)


def _strip_footers(text: str, title: str | None) -> str:
    """Remove the repeating Indian Kanoon page furniture."""
    text = _KANOON_URL.sub("", text)
    if title:
        # The title line repeats on every page, sometimes truncated with "...".
        stem = re.escape(title[:40])
        text = re.sub(rf"^{stem}[^\n]*on \d{{1,2}} \w+, \d{{4}}\s*$", "", text, flags=re.M)
    # Bare page numbers left behind on their own line.
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.M)
    # Collapse the blank lines all that stripping created.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_pdf(path: Path) -> dict:
    """Extract text + header metadata from one judgment PDF."""
    doc = pymupdf.open(path)
    raw = "".join(page.get_text() for page in doc)
    n_pages = doc.page_count
    doc.close()

    # The header block ends at "Author:"/"Bench:" if present. Collapse its
    # whitespace first -- both the title and the date wrap across lines.
    head_raw = raw[:1500]
    cut = min(
        (i for i in (head_raw.find("Author:"), head_raw.find("Bench:")) if i > 0),
        default=400,
    )
    head_flat = " ".join(head_raw[:cut].split())

    title_m = _HEADER_TITLE.search(head_flat)
    title = title_m.group(1).strip(" ,-") if title_m else path.stem
    decided_on = " ".join(title_m.group(2).split()) if title_m else None

    url_m = _KANOON_URL.search(raw)
    source_url = f"http://indiankanoon.org/doc/{url_m.group(1)}/" if url_m else None

    author_m, bench_m = _AUTHOR.search(head_raw), _BENCH.search(head_raw)
    # Court appears in the body, and often wraps -- search flattened text.
    court_m = _COURT.search(" ".join(raw[:6000].split()))

    text = _strip_footers(raw, title)

    return {
        "doc_id": path.stem,
        "title": title,
        "decided_on": decided_on,
        "author": author_m.group(1).strip() if author_m else None,
        "bench": bench_m.group(1).strip() if bench_m else None,
        "court": " ".join(court_m.group(1).split()).title() if court_m else None,
        "source_url": source_url,
        "text": text,
        "n_pages": n_pages,
        "n_chars": len(text),
    }


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a judgment into (section_name, body) pairs.

    We only split on markers we can trust. Everything before the first marker is
    'preamble' (cause title, parties, counsel); everything after the last is
    carried under that marker. Over-segmenting is worse than under-segmenting --
    a wrong boundary silently truncates a ratio mid-sentence.
    """
    cuts: list[tuple[int, str]] = []
    for name, rx in _SECTION_MARKERS:
        for m in rx.finditer(text):
            cuts.append((m.start(), name))
    if not cuts:
        return [("body", text)]

    cuts.sort()
    out: list[tuple[str, str]] = []
    if cuts[0][0] > 0:
        out.append(("preamble", text[: cuts[0][0]].strip()))
    for i, (pos, name) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
        body = text[pos:end].strip()
        if body:
            out.append((name, body))
    return out


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character chunking that prefers paragraph, then sentence, boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer a paragraph break in the last 30% of the window.
            br = window.rfind("\n\n", int(size * 0.7))
            if br == -1:
                br = window.rfind(". ", int(size * 0.7))
            if br != -1:
                end = start + br + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(parsed: dict) -> list[Chunk]:
    """Structure-aware chunks carrying a contextual header.

    Each chunk is prefixed with case/court/section context so it can be embedded
    and read standalone -- 'contextual retrieval'. Without this a passage saying
    "the appeal is allowed" is meaningless in isolation.
    """
    out: list[Chunk] = []
    for section, body in split_sections(parsed["text"]):
        for i, piece in enumerate(chunk_text(body, settings.chunk_chars, settings.chunk_overlap)):
            para = _PARA_NUM.search(piece)
            parts = [f"Case: {parsed['title']}"]
            if parsed["court"]:
                parts.append(f"Court: {parsed['court']}")
            if parsed["decided_on"]:
                parts.append(f"Decided: {parsed['decided_on']}")
            parts.append(f"Section: {section}")
            header = " | ".join(parts)
            out.append(
                Chunk(
                    chunk_id=f"{parsed['doc_id']}::{section}::{i}",
                    doc_id=parsed["doc_id"],
                    section=section,
                    para_start=para.group(1) if para else None,
                    text=piece,
                    context_header=header,
                )
            )
    return out


def ingest_corpus(corpus_dir: Path | None = None) -> tuple[list[dict], list[Chunk]]:
    """Parse and chunk every PDF in the corpus."""
    corpus_dir = corpus_dir or settings.corpus_dir
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {corpus_dir}")

    docs, chunks = [], []
    for p in pdfs:
        parsed = parse_pdf(p)
        docs.append(parsed)
        chunks.extend(build_chunks(parsed))
    return docs, chunks


def main() -> None:
    docs, chunks = ingest_corpus()
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=1, ensure_ascii=False)
    )
    print(f"Parsed {len(docs)} judgments -> {len(chunks)} chunks")
    print(f"  chars: {sum(d['n_chars'] for d in docs):,}  pages: {sum(d['n_pages'] for d in docs)}")
    print(f"  wrote {settings.chunks_path}")


if __name__ == "__main__":
    main()
