"""Bake the embedding model into the image at build time.

Without this the first request pays a 1.1 GB download and ~12 s of load. Run
once during build; TRANSFORMERS_OFFLINE then stops the runtime reaching for the
network at all.
"""
from sentence_transformers import SentenceTransformer

SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
print("embedding model cached")
