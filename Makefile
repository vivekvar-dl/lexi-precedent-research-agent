.PHONY: install ingest enrich index build gold evals evals-full test app clean

export PYTHONPATH := src
PY := .venv/bin/python

install:
	python3 -m venv .venv
	.venv/bin/pip install --resume-retries 10 -r requirements.txt

# --- index pipeline (run once) ------------------------------------------------
ingest:
	$(PY) -m lexi.ingest

enrich:            ## 56 LLM calls; resumable -- safe to re-run after a quota stop
	$(PY) -m lexi.enrich

index:
	$(PY) -m lexi.index

build: ingest enrich index

# --- evaluation ---------------------------------------------------------------
gold:              ## build the 56 x 15 gold label set
	$(PY) -m evals.gold

evals:             ## run agent over all queries, score all four dimensions
	$(PY) -m evals.run_all

evals-full:        ## adds self-consistency across 3 seeds
	$(PY) -m evals.run_all --seeds 3

test:
	$(PY) -m pytest evals/ -v

# --- app ----------------------------------------------------------------------
app:
	.venv/bin/streamlit run app.py

clean:
	rm -rf index/lance evals/runs evals/results
