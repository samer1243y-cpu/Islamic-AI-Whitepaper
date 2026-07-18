# Islamic Retrieval Engine

This project provides a local offline hybrid retrieval engine for Arabic Islamic content.

## What it supports

- Hybrid retrieval using BM25 and semantic search
- Arabic normalization (`normalize_arabic`) and right-to-left terminal display
- Automatic loading of JSONL/JSON sources from `processed/quran`, `processed/tafsir`, `processed/hadith`, `processed/fatwas`
- Automatic JSON -> JSONL conversion for files in `processed/*`
- Progress bars during indexing
- Persistent indexes in `indexes/`
- Debug logging in `logs/errors.log`
- Local offline execution only; no paid APIs used

## Installation

1. Open the workspace in VS Code.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the engine

```bash
python retrieval_engine.py
```

If the indexes already exist in `indexes/`, they are loaded directly. If data changes, delete `indexes/` to force a rebuild.

## Adding new sources

Place new `.jsonl` or `.json` files inside one of these directories:

- `processed/quran`
- `processed/tafsir`
- `processed/hadith`
- `processed/fatwas`

Each record should include at least `clean_text`, and ideally also:

- `source_type`
- `scholar`
- `title`
- `surah`
- `ayah_range`

JSON files are converted automatically to JSONL where possible.

## How retrieval works

The engine uses two retrieval signals:

- **BM25**: fast keyword-based ranking. Good for exact phrase matching and surface relevance.
- **Semantic search**: meaning-based similarity using a multilingual embedding model. Good for related concepts and paraphrases.

The final ranking is a weighted combination:

- BM25 weight = 0.7
- Semantic weight = 0.3

This hybrid approach improves relevance for Arabic queries while keeping the search stable and interpretable.

## Future expansion

This engine is designed for future growth toward:

- Agentic RAG
- Multi-agent routing
- LLM grounded answers

## Sample queries

- `ما حكم الإسبال`
- `ما معنى الصراط المستقيم`
- `حديث عن الربا`
- `حكم حلق اللحية`
