# Lexi — legal precedent research agent
#
# The image bakes in the prebuilt LanceDB index and the embedding model, so the
# container starts without making a single LLM call and without downloading
# 1.1 GB on first request.
#
# Measured sizing (see README "Deploying it"): the embedding model alone holds
# 1,427 MB resident once loaded, so the host needs >= 2 GB. A 512 MB tier OOMs
# on model load -- that is the first thing to get right, and no code tuning
# avoids it.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # Keep HF caches inside the image rather than a writable layer at runtime.
    HF_HOME=/app/.hf \
    LEXI_EMBED_DEVICE=cpu

WORKDIR /app

# System deps: pymupdf needs libgl/glib at import time even when only reading
# already-parsed text.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# --- bake the embedding model into the image ---------------------------------
# Without this the first request pays a 1.1 GB download and ~12 s of load. The
# download happens once, at build time; the offline flags set immediately after
# stop the runtime from ever reaching for the network.
COPY docker/fetch_model.py /tmp/fetch_model.py
RUN python /tmp/fetch_model.py

# Offline only from here on. Set in the global ENV block it also applied during
# the build, so the download step above could not reach huggingface.co and the
# image failed to build. At runtime it is what turns a cache miss into a loud
# error instead of a silent 1.1 GB download on a user's first request.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY src/ ./src/
COPY app.py ./
COPY index/ ./index/
COPY docker/smoke_test.py ./docker/smoke_test.py

# --- build the vector store if it was not committed ---------------------------
# `index/lance/` is git-ignored (it is regenerated), while `case_cards.json` and
# `chunks.json` ARE committed. This re-embeds from those committed chunks and
# needs ZERO LLM calls -- enrichment already happened offline.
COPY docker/build_index.py /tmp/build_index.py
RUN python /tmp/build_index.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/_stcore/health || exit 1

# Streamlit binds 8000 to match Azure App Service's default WEBSITES_PORT.
# CORS/XSRF are disabled because the platform terminates TLS in front of us and
# the app holds no session state worth forging.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
