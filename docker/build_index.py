"""Build the LanceDB store at image-build time if it was not committed.

`index/lance/` is git-ignored (it is regenerated), while `case_cards.json` and
`chunks.json` ARE committed. This re-embeds from those committed chunks and
makes ZERO LLM calls -- enrichment already happened offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

if Path("index/lance").exists():
    print("index already present, skipping build")
else:
    from lexi.index import build_index

    build_index()
    print("index built")
