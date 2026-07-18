import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ============================================
# Arabic Text Fix (RTL Support)
# ============================================

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    RTL_AVAILABLE = True

except ImportError:
    RTL_AVAILABLE = False

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_PATH = "processed/tafsir/tafsir_14_Tafsir_Ibn_Kathir_arabic.processed.jsonl"

TOP_K = 5

# ============================================
# RTL FORMATTER
# ============================================

def format_arabic(text):

    if not RTL_AVAILABLE:
        return text

    reshaped = arabic_reshaper.reshape(text)

    return get_display(reshaped)

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")

# ============================================
# LOAD DATA
# ============================================

records = []
texts = []

print("\nLoading processed tafsir data...")

with open(DATA_PATH, "r", encoding="utf-8") as f:

    for line in f:

        obj = json.loads(line)

        text = obj.get("embedding_text", "").strip()

        if not text:
            continue

        records.append(obj)
        texts.append(text)

print(f"Loaded {len(records)} records.")

# ============================================
# BM25 INDEX
# ============================================

print("\nBuilding BM25 index...")

tokenized_corpus = [text.split() for text in texts]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")

# ============================================
# GENERATE EMBEDDINGS
# ============================================

print("\nGenerating embeddings in batches...")

batch_size = 32

all_embeddings = []

for i in range(0, len(texts), batch_size):

    batch = texts[i:i + batch_size]

    batch_embeddings = model.encode(
        batch,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.extend(batch_embeddings)

    print(f"Processed {i + len(batch)} / {len(texts)}")

embeddings = np.array(all_embeddings).astype("float32")

print("\nEmbeddings generated successfully.")

# ============================================
# BUILD FAISS INDEX
# ============================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS semantic index ready.")

# ============================================
# HYBRID SEARCH
# ============================================

def hybrid_search(query):

    # ========================================
    # BM25 SEARCH
    # ========================================

    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(bm25_scores)[::-1][:20]

    # ========================================
    # SEMANTIC SEARCH
    # ========================================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        20
    )

    # ========================================
    # MERGE RESULTS
    # ========================================

    combined_scores = {}

    # BM25 weight
    for idx in bm25_top_indices:

        combined_scores[idx] = combined_scores.get(idx, 0) + (
            bm25_scores[idx] * 0.6
        )

    # Semantic weight
    for i, idx in enumerate(semantic_indices[0]):

        combined_scores[idx] = combined_scores.get(idx, 0) + (
            semantic_scores[0][i] * 0.4
        )

    # Sort
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:TOP_K]

# ============================================
# SEARCH LOOP
# ============================================

print("\n" + "=" * 70)
print("Islamic Hybrid Retrieval System Ready")
print("=" * 70)

while True:

    print("\n" + "=" * 70)

    query = input("\nEnter your query (or 'exit'): ").strip()

    if query.lower() == "exit":
        break

    results = hybrid_search(query)

    print("\nTOP RESULTS:\n")

    for rank, (idx, score) in enumerate(results, start=1):

        result = records[idx]

        print("=" * 90)

        print(f"\nRank: {rank}")
        print(f"Combined Score: {score:.4f}")

        metadata = (
            f"Surah: {result.get('surah')} | "
            f"Ayah Range: {result.get('ayah_range')}"
        )

        print(format_arabic(metadata))

        print("\nTEXT:\n")

        text = result.get("clean_text", "")[:2000]

        print(format_arabic(text))

        print("\n")

print("\nSystem exited.")