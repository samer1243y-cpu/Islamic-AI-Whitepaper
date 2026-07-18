"""Quick validation script for the Islamic RAG pipeline.

This script uses the existing retrieval engine and answer generator.
It does not rebuild indexes, generate embeddings, or write any reports.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from answer_generator import AnswerGenerator
from data_pipeline import load_processed_records
from retrieval_engine import search, setup_logger
from config import BM25_CACHE_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE

TEST_QUERIES = [
    "ما حكم الإسبال؟",
    "حديث عن بر الوالدين",
    "ما معنى الصراط المستقيم؟",
    "آية عن الصبر",
    "من هو رئيس الولايات المتحدة؟",
]

EXPECTED_TYPE_MAP = {
    "ما حكم الإسبال؟": "fatwa",
    "حديث عن بر الوالدين": "hadith",
    "ما معنى الصراط المستقيم؟": "tafsir",
    "آية عن الصبر": "quran",
    "من هو رئيس الولايات المتحدة؟": "refusal",
}

REFUSAL_KEYWORDS = [
    "لم أجد جواباً صريحاً",
    "تعذر توليد الإجابة",
    "لا يوجد",
    "لا يمكن",
    "غير موجود",
    "لا أستطيع",
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def build_tokenized_texts(records: list[dict]) -> list[list[str]]:
    tokenized = []
    pattern = re.compile(r"[\u0600-\u06FF]+|[A-Za-z0-9]+")
    for record in records:
        text = normalize_text(record.get("clean_text", record.get("text", "")))
        tokenized.append(pattern.findall(text))
    return tokenized


def top_source_types(retrieved_docs: list[dict]) -> list[str]:
    counts = Counter()
    for record in retrieved_docs:
        source_type = (record.get("source_type") or "").strip().lower()
        if source_type:
            counts[source_type] += 1
    return [source for source, _ in counts.most_common(3)]


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().lower()
    return any(keyword in normalized for keyword in REFUSAL_KEYWORDS)


def print_test_header(query: str) -> None:
    print("\n" + "#" * 80)
    print(f"Query: {query}")
    print("#" * 80)


def main() -> None:
    logger = setup_logger(False)

    missing = [path for path in (EMBEDDINGS_FILE, FAISS_INDEX_FILE, BM25_CACHE_FILE) if not path.exists()]
    if missing:
        print("ERROR: One or more index artifacts are missing. This script will not rebuild indexes.")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    records = load_processed_records(logger)
    if not records:
        print("ERROR: No processed records found. Ensure processed sources exist under processed/.")
        sys.exit(1)

    try:
        with open(BM25_CACHE_FILE, "rb") as fh:
            bm25 = pickle.load(fh)
    except Exception as exc:
        print(f"ERROR: Failed to load BM25 index: {exc}")
        sys.exit(1)

    try:
        index = faiss.read_index(str(FAISS_INDEX_FILE))
    except Exception as exc:
        print(f"ERROR: Failed to load FAISS index: {exc}")
        sys.exit(1)

    tokenized_texts = build_tokenized_texts(records)
    reranker = None

    try:
        generator = AnswerGenerator()
    except Exception as exc:
        print(f"ERROR: Failed to initialize AnswerGenerator: {exc}")
        sys.exit(1)

    total = 0
    passed = 0
    failed = 0

    for query in TEST_QUERIES:
        total += 1
        expected_type = EXPECTED_TYPE_MAP.get(query, "")

        print_test_header(query)

        results = search(query, records, tokenized_texts, bm25, index, reranker, top_k=10)
        retrieved_docs = [records[idx] for idx, _ in results]
        source_types = top_source_types(retrieved_docs)
        print(f"Top source types returned: {', '.join(source_types) if source_types else 'none'}")

        answer_data = generator.generate_answer(query, retrieved_docs)
        print(f"Generated answer:\n{answer_data['answer']}\n")
        print(f"Verification status: {answer_data['verified']}")

        if expected_type == "refusal":
            passed_test = answer_data["verified"] is False or is_refusal(answer_data["answer"])
        else:
            passed_type = expected_type in source_types
            passed_test = passed_type and answer_data["verified"] is True

        if passed_test:
            passed += 1
            print("Test result: PASS")
        else:
            failed += 1
            print("Test result: FAIL")

    print("\n" + "=" * 80)
    print(f"Total tests: {total}")
    print(f"Passed tests: {passed}")
    print(f"Failed tests: {failed}")


if __name__ == "__main__":
    main()
