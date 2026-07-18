import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENABLE_RERANKER", "false")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import retrieval_engine as engine
from indexer import load_or_build_index


def main() -> None:
    logger = engine.setup_logger(False)
    engine.ensure_directories()

    records = engine.load_processed_records(logger)
    if not records:
        logger.error("No processed records found. Run data_pipeline.py first.")
        return

    index, embeddings, tokenized_texts, bm25 = load_or_build_index(records, rebuild=False, logger=logger)
    reranker = engine.load_reranker()

    queries = [
        "ما حكم الإسبال",
        "حديث عن الربا",
        "ما معنى الصراط المستقيم",
        "هل يجوز حلق اللحية",
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print("=" * 100)
        results = engine.search(query, records, tokenized_texts, bm25, index, reranker, top_k=5)
        if not results:
            print("No results found.")
            continue
        for rank, (idx, score) in enumerate(results, start=1):
            record = records[idx]
            engine.print_result(rank, score, record)
        print("=" * 100)


if __name__ == "__main__":
    main()
