# NCERT RAG Bot

NCERT RAG Bot is an experimental local question-answering application for studying Class 10 NCERT material with retrieval-augmented generation (RAG).

## Status

The repository contains a Streamlit implementation plus additional frontend/backend experiments. The root Python workflow is the clearest starting point. Generated vector databases, uploaded PDFs, extracted textbook text, virtual environments, and Python caches are intentionally excluded from version control.

This is an educational prototype. Answers may be incomplete or incorrect and should be checked against the official textbook.

## How it works

```text
NCERT PDF or extracted text
          |
          v
      text chunks
          |
          v
 Chroma vector index
          |
          v
 relevant passages + question
          |
          v
    local Ollama model
          |
          v
 answer grounded in retrieved context
```

## Root application

- `data_ingestion.py` — prepares source text for indexing.
- `chunking.py` — splits source material into retrieval chunks.
- `rag_pipeline.py` — builds the retrieval and answer pipeline.
- `streamlit_app.py` — provides the interactive chat interface.
- `Modelfile` — local Ollama model configuration.
- `requirements.txt` — pinned Python dependencies for the root prototype.

Other directories contain separate full-stack experiments and are not required to run the root Streamlit application.

## Requirements

- Python 3.11 or another version compatible with the pinned dependencies
- Ollama
- An NCERT source document obtained through an authorized source

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Ollama using its official instructions, then make the model referenced by the local configuration available.

Do not commit textbook PDFs, extracted textbook content, uploaded documents, or generated vector indexes. Place local source material in the path expected by the application and run the ingestion workflow before launching the UI.

```bash
python data_ingestion.py
streamlit run streamlit_app.py
```

## Evaluation status

The repository does not currently provide a versioned labeled evaluation set or verified benchmark results. Before presenting accuracy claims, add:

- retrieval recall at a defined value of `k`
- citation/source accuracy
- groundedness or faithfulness checks
- answer-quality comparison against a documented baseline
- exact model, dataset revision, parameters, and random seeds

## Data and licensing

NCERT materials must be obtained and used according to their applicable terms. This repository does not redistribute the previously local textbook PDF, extracted text, or generated index. Cite the official textbook edition and retrieval date when publishing results.

## Limitations

- Generated answers are not authoritative educational guidance.
- Retrieval quality depends on source extraction and chunking.
- Local model behavior varies by model version and configuration.
- The repository contains multiple experimental application structures that still need consolidation.

## Next steps

- Consolidate one canonical application structure.
- Add a small redistributable labeled evaluation fixture.
- Add tests for chunking, retrieval, and prompt construction.
- Add CI for linting, tests, and a lightweight ingestion smoke test.
- Document measured results only after reproducing them.
