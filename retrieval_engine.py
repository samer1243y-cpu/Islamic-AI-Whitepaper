import argparse
import logging
import re
import sys
import os
import textwrap
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_ENABLED = True
except ImportError:
    RTL_ENABLED = False

from config import (
    BM25_WEIGHT,
    HYBRID_TOP_K,
    INDEX_DIR,
    LOG_DIR,
    MODEL_NAME,
    RE_RANK_TOP_K,
    SEMANTIC_WEIGHT,
    TOP_K,
    WEAK_SCORE_THRESHOLD,
)
from data_pipeline import load_processed_records
from indexer import load_or_build_index

logger = None
PRIORITY_BOOST = 0.15


def setup_logger(debug: bool) -> logging.Logger:
    global logger
    logger = logging.getLogger("RetrievalEngine")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "errors.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(console_handler)

    return logger


def format_arabic(text: str) -> str:
    if not text:
        return text
    if RTL_ENABLED:
        try:
            shaped = arabic_reshaper.reshape(text)
            return get_display(shaped)
        except Exception:
            return text
    return text


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u08D3-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F\u200E\u200F\u202A-\u202E]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[\u0600-\u06FF]+|[A-Za-z0-9]+", text)
    return [token for token in tokens if token.strip()]


def ensure_directories() -> None:
    for directory in [INDEX_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


model = None

def get_model(model_name: str = MODEL_NAME):
    global model
    if model is None:
        model = SentenceTransformer(model_name)
    return model


def load_reranker() -> object | None:
    # Avoid auto-downloading large reranker models in CPU environments.
    # Enable explicitly with env var ENABLE_RERANKER=true
    enable = str(os.environ.get("ENABLE_RERANKER", "false")).lower() in ("1", "true", "yes")
    if not enable:
        logger.info("Reranker disabled via ENABLE_RERANKER env var")
        return None

    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        logger.warning(f"CrossEncoder unavailable: {exc}")
        return None

    try:
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
        logger.info(f"Loaded reranker on {device}")
        return reranker
    except Exception as exc:
        logger.warning(f"Failed to load reranker model: {exc}")
        return None


def route_query(query: str) -> str:
    q = query.lower()
    if any(token in q for token in ["ما حكم", "هل يجوز", "حكم"]):
        return "hukm"
    if any(token in q for token in ["ما معنى", "تفسير"]):
        return "tafsir"
    if "حديث" in q:
        return "hadith"
    return "default"


def get_priority_boost(source_type: str, mode: str) -> float:
    priorities = {
        "hukm": ["fatwa", "hadith"],
        "tafsir": ["tafsir", "quran"],
        "hadith": ["hadith"],
    }
    return PRIORITY_BOOST if source_type in priorities.get(mode, []) else 0.0


def rerank_candidates(query: str, candidate_ids: list[int], records: list[dict], reranker: object) -> list[tuple[int, float]]:
    if reranker is None or not candidate_ids:
        return [(idx, 0.0) for idx in candidate_ids]

    inputs = [(query, records[idx].get("clean_text", "")) for idx in candidate_ids]
    try:
        scores = reranker.predict(inputs, batch_size=16)
        return [(idx, float(score)) for idx, score in zip(candidate_ids, scores)]
    except Exception as exc:
        logger.warning(f"Reranker prediction failed: {exc}")
        return [(idx, 0.0) for idx in candidate_ids]


def search(query: str, records: list[dict], tokenized_texts: list[list[str]], bm25: BM25Okapi, index: faiss.IndexFlatIP, reranker: object, top_k: int) -> list[tuple[int, float]]:
    query_norm = normalize_arabic(query)
    query_tokens = tokenize(query_norm)
    if query_tokens:
        bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
    else:
        bm25_scores = np.zeros(len(records), dtype=np.float32)

    bm25_top = np.argsort(bm25_scores)[::-1][:HYBRID_TOP_K]
    # Per E5 prompt-tuning requirement, prefix queries with 'query: '
    query_for_embedding = "query: " + query_norm
    model_local = get_model()
    query_embedding = model_local.encode([query_for_embedding], normalize_embeddings=True, show_progress_bar=False)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    semantic_scores, semantic_ids = index.search(query_embedding, HYBRID_TOP_K)
    semantic_scores = np.nan_to_num(semantic_scores[0], nan=0.0, posinf=0.0, neginf=0.0)
    semantic_ids = semantic_ids[0]

    combined: dict[int, float] = {}
    for idx in bm25_top:
        combined[idx] = combined.get(idx, 0.0) + float(bm25_scores[idx]) * BM25_WEIGHT
    for idx, score in zip(semantic_ids, semantic_scores):
        if idx < 0 or idx >= len(records):
            continue
        combined[idx] = combined.get(idx, 0.0) + float(score) * SEMANTIC_WEIGHT

    mode = route_query(query)
    for idx in list(combined.keys()):
        if 0 <= idx < len(records):
            combined[idx] += get_priority_boost(records[idx].get("source_type", ""), mode)

    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:HYBRID_TOP_K]
    candidate_ids = [idx for idx, _ in ranked]

    if reranker is not None and candidate_ids:
        reranked = rerank_candidates(query_norm, candidate_ids[:RE_RANK_TOP_K], records, reranker)
        reranked = sorted(reranked, key=lambda item: item[1], reverse=True)
        return reranked[:top_k]

    return ranked[:top_k]


def safe_print(text: str, width: int = 100) -> None:
    if text is None:
        text = ""
    wrapped = textwrap.fill(text, width=width)
    try:
        print(wrapped)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write(wrapped.encode("utf-8", errors="replace") + b"\n")
        except Exception:
            print(wrapped.encode("utf-8", errors="replace"))


def print_result(rank: int, score: float, record: dict) -> None:
    separator = "=" * 100
    print(separator)
    print(f"Rank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Source Type: {record.get('source_type', '')}")
    print(f"Book / Scholar: {record.get('book', '') or record.get('scholar', '')}")
    print(f"Surah / Hadith: {record.get('surah', '') or record.get('chapter', '') or record.get('hadith_number', '')}")
    print(separator)
    print("TEXT:")
    safe_print(format_arabic(record.get("clean_text", record.get("text", ""))), width=100)
    print(separator)


def query_loop(records: list[dict], tokenized_texts: list[list[str]], bm25: BM25Okapi, index: faiss.IndexFlatIP, reranker: object, top_k: int) -> None:
    print("\n" + "=" * 70)
    print("Islamic Local Retrieval Engine Ready")
    print("=" * 70)
    print("Suggested queries:")
    print("- ما حكم الإسبال")
    print("- ما معنى الصراط المستقيم")
    print("- حديث عن الربا")
    print("- حكم حلق اللحية")

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search(query, records, tokenized_texts, bm25, index, reranker, top_k)
        if not results:
            print("No results found. Corpus coverage may be insufficient.")
            continue

        if results[0][1] < WEAK_SCORE_THRESHOLD:
            print("Corpus coverage may be insufficient.")

        for rank, (idx, score) in enumerate(results, start=1):
            print_result(rank, score, records[idx])

    print("\nSystem exited.")


def main() -> None:
    global model
    parser = argparse.ArgumentParser(description="Run the Islamic Local Retrieval Engine.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild the FAISS and BM25 indexes")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    args = parser.parse_args()

    setup_logger(args.debug)
    ensure_directories()

    if not RTL_ENABLED:
        logger.warning("arabic_reshaper or python-bidi not installed. Arabic terminal support may be degraded.")

    records = load_processed_records(logger)
    if not records:
        logger.error("No records found in processed data directories. Please add JSONL sources under processed/*.")
        print("No data sources available. Check logs/errors.log for details.")
        return

    model = SentenceTransformer(MODEL_NAME)
    index, embeddings, tokenized_texts, bm25 = load_or_build_index(records, args.rebuild_index, logger)
    reranker = load_reranker()

    query_loop(records, tokenized_texts, bm25, index, reranker, args.top_k)


if __name__ == "__main__":
    main()
