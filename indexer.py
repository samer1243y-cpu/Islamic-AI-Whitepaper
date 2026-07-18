import argparse
import json
import logging
import os
import pickle
import time
from hashlib import sha256
from pathlib import Path

import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    BM25_CACHE_FILE,
    BATCH_SIZE,
    OUTPUT_PATHS,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    INDEX_DIR,
    INDEX_METADATA_FILE,
    MAX_MEMORY_BYTES,
    MIN_BATCH_SIZE,
    MODEL_NAME,
    RAW_SOURCE_DIRS,
    PROCESSED_SOURCE_DIRS,
)
from data_pipeline import gather_source_files, load_processed_records
from rank_bm25 import BM25Okapi


def ensure_directories() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("Indexer")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def compute_source_signature(paths: list[Path]) -> str:
    parts = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except FileNotFoundError:
            continue
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def choose_batch_size(num_records: int, embed_dim: int) -> int:
    batch_size = min(BATCH_SIZE, 32)
    estimated = batch_size * embed_dim * np.dtype(np.float16).itemsize
    while batch_size > MIN_BATCH_SIZE and estimated > MAX_MEMORY_BYTES:
        batch_size //= 2
        estimated = batch_size * embed_dim * np.dtype(np.float16).itemsize
    return max(batch_size, MIN_BATCH_SIZE)


def load_metadata() -> dict:
    if not INDEX_METADATA_FILE.exists():
        return {}
    try:
        return json.loads(INDEX_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_metadata(metadata: dict) -> None:
    INDEX_METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_embeddings(texts: list[str], model: SentenceTransformer, logger: logging.Logger) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")

    example = model.encode([texts[0]], normalize_embeddings=True, show_progress_bar=False)
    embed_dim = int(np.asarray(example).shape[1])
    batch_size = choose_batch_size(len(texts), embed_dim)
    dtype = np.float16

    logger.info(f"Building embeddings with model={MODEL_NAME}, batch_size={batch_size}, dtype={dtype}")
    embeddings = np.zeros((len(texts), embed_dim), dtype=dtype)
    start_time = time.perf_counter()
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches", unit="batch"):
        slice_text = texts[start : start + batch_size]
        batch_embeddings = model.encode(slice_text, normalize_embeddings=True, show_progress_bar=False)
        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        embeddings[start : start + len(slice_text)] = batch_embeddings.astype(dtype)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Finished embeddings in {elapsed:.2f}s")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings.astype(np.float32))
    return index


def build_bm25(tokenized_texts: list[list[str]], logger: logging.Logger) -> BM25Okapi:
    logger.info("Building BM25 index")
    bm25 = BM25Okapi(tokenized_texts)
    return bm25


def save_bm25(bm25: BM25Okapi) -> None:
    with open(BM25_CACHE_FILE, "wb") as fh:
        pickle.dump(bm25, fh)


def load_bm25(logger: logging.Logger) -> BM25Okapi | None:
    if not BM25_CACHE_FILE.exists():
        return None
    try:
        with open(BM25_CACHE_FILE, "rb") as fh:
            bm25 = pickle.load(fh)
        return bm25
    except Exception as exc:
        logger.warning(f"Failed to load BM25 cache: {exc}")
        return None


def load_or_build_index(records: list[dict], rebuild: bool, logger: logging.Logger, model_name: str = MODEL_NAME) -> tuple[faiss.IndexFlatIP, np.ndarray, list[list[str]], BM25Okapi]:
    texts = [record.get("embedding_text", "") for record in records]
    tokenized_texts = [re.findall(r"[\u0600-\u06FF]+|[A-Za-z0-9]+", normalize_text(record.get("clean_text", ""))) for record in records]

    metadata = load_metadata()
    cached_model = metadata.get("model_name")

    # If cached artifacts exist and model matches (and not rebuild), load them.
    if not rebuild and EMBEDDINGS_FILE.exists() and FAISS_INDEX_FILE.exists() and cached_model == model_name:
        try:
            embeddings = np.load(EMBEDDINGS_FILE)
            index = faiss.read_index(str(FAISS_INDEX_FILE))
            if index.ntotal != len(records) or embeddings.shape[0] != len(records):
                logger.warning("Loaded cached index size differs from processed records; continuing with cached index.")
            bm25 = load_bm25(logger)
            if bm25 is None:
                logger.warning("BM25 cache missing, rebuilding BM25 from processed records.")
                bm25 = build_bm25(tokenized_texts, logger)
                save_bm25(bm25)
                logger.info(f"Saved BM25 cache to {BM25_CACHE_FILE}")
            else:
                logger.info("Loaded cached embeddings, FAISS index, and BM25 index.")
            logger.info(f"Embedding dims: {embeddings.shape[1]}, total vectors: {embeddings.shape[0]}")
            try:
                import psutil
                proc = psutil.Process()
                logger.info(f"Memory usage: {proc.memory_info().rss / (1024**2):.2f} MB RSS")
            except Exception:
                logger.info("Memory usage: psutil not installed; skipping RSS report")
            return index, embeddings, tokenized_texts, bm25
        except Exception as exc:
            logger.warning(f"Failed to load cached artifacts: {exc}; rebuilding index.")

    # If cached artifacts exist but model differs, force rebuild to ensure compatible embeddings.
    if EMBEDDINGS_FILE.exists() and FAISS_INDEX_FILE.exists() and cached_model != model_name:
        logger.info(f"Existing embeddings were created with '{cached_model}'; rebuilding embeddings with '{model_name}'.")
        try:
            EMBEDDINGS_FILE.unlink()
            FAISS_INDEX_FILE.unlink()
            logger.info("Removed old embeddings and FAISS index to force rebuild.")
        except Exception as exc:
            logger.warning(f"Failed to remove old index files: {exc}; continuing and will overwrite if possible.")

    # Prepare texts by prefixing the required passage prompt per E5 requirement.
    texts_for_embedding = ["passage: " + (t or "") for t in texts]

    model = SentenceTransformer(model_name)
    start_time = time.perf_counter()
    embeddings = build_embeddings(texts_for_embedding, model, logger)
    index = build_faiss_index(embeddings)
    elapsed = time.perf_counter() - start_time

    # Save artifacts
    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE} and FAISS index to {FAISS_INDEX_FILE}")

    bm25 = build_bm25(tokenized_texts, logger)
    save_bm25(bm25)
    logger.info(f"Saved BM25 cache to {BM25_CACHE_FILE}")

    embedding_size = EMBEDDINGS_FILE.stat().st_size if EMBEDDINGS_FILE.exists() else 0
    faiss_size = FAISS_INDEX_FILE.stat().st_size if FAISS_INDEX_FILE.exists() else 0
    bm25_size = BM25_CACHE_FILE.stat().st_size if BM25_CACHE_FILE.exists() else 0
    logger.info(f"Embedding dims: {embeddings.shape[1]}, total vectors: {embeddings.shape[0]}")
    logger.info(f"File sizes: embeddings={embedding_size} bytes, faiss={faiss_size} bytes, bm25={bm25_size} bytes")
    logger.info(f"Indexing elapsed time: {elapsed:.2f}s")
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        logger.info(f"Memory usage: {proc.memory_info().rss / (1024**2):.2f} MB RSS")
    except Exception:
        logger.info("Memory usage: psutil not installed; skipping RSS report")

    save_metadata({"record_count": len(records), "model_name": model_name})
    return index, embeddings, tokenized_texts, bm25

    embedding_size = EMBEDDINGS_FILE.stat().st_size if EMBEDDINGS_FILE.exists() else 0
    faiss_size = FAISS_INDEX_FILE.stat().st_size if FAISS_INDEX_FILE.exists() else 0
    bm25_size = BM25_CACHE_FILE.stat().st_size if BM25_CACHE_FILE.exists() else 0
    logger.info(f"Embedding dims: {embeddings.shape[1]}, total vectors: {embeddings.shape[0]}")
    logger.info(f"File sizes: embeddings={embedding_size} bytes, faiss={faiss_size} bytes, bm25={bm25_size} bytes")
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        logger.info(f"Memory usage: {proc.memory_info().rss / (1024**2):.2f} MB RSS")
    except ImportError:
        logger.info("Memory usage: psutil not installed, unable to report RSS")

    save_metadata({"signature": current_signature, "record_count": len(records)})
    return index, embeddings, tokenized_texts, bm25


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh local FAISS and BM25 indexes.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild the index from processed data")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Embedding model name or path")
    args = parser.parse_args()

    logger = setup_logger(args.debug)
    ensure_directories()

    records = load_processed_records(logger)
    if not records:
        logger.error("No processed records found. Run data_pipeline.py first.")
        return

    index, embeddings, tokenized_texts, bm25 = load_or_build_index(records, args.rebuild_index, logger, model_name=args.model)
    logger.info(f"Index ready with {len(records)} records.")


if __name__ == "__main__":
    main()
