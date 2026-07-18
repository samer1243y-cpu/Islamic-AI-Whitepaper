import os
import json
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =====================================================
# RTL Support
# =====================================================

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    RTL_AVAILABLE = True

except:
    RTL_AVAILABLE = False

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_DIRECTORIES = [
    "processed/quran",
    "processed/tafsir",
    "processed/hadith",
    "processed/fatwas"
]

TOP_K = 5

# =====================================================
# RTL Formatter
# =====================================================

def format_arabic(text):

    if not RTL_AVAILABLE:
        return text

    reshaped = arabic_reshaper.reshape(text)

    return get_display(reshaped)

# =====================================================
# Arabic Normalization
# =====================================================

def normalize_arabic(text):

    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)

    text = re.sub(r'[إأآا]', 'ا', text)

    text = re.sub(r'ى', 'ي', text)

    text = re.sub(r'ؤ', 'و', text)

    text = re.sub(r'ئ', 'ي', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")

# =====================================================
# LOAD RECORDS
# =====================================================

records = []
texts = []

print("\nLoading all Islamic sources...")

for directory in DATA_DIRECTORIES:

    if not os.path.exists(directory):
        continue

    for filename in os.listdir(directory):

        if not filename.endswith(".jsonl"):
            continue

        path = os.path.join(directory, filename)

        print(f"Loading: {path}")

        with open(path, "r", encoding="utf-8") as f:

            for line in f:

                try:

                    obj = json.loads(line)

                    base_text = obj.get("clean_text", "").strip()

                    if not base_text:
                        continue

                    source_type = obj.get(
                        "source_type",
                        "unknown"
                    )

                    scholar = obj.get(
                        "scholar",
                        ""
                    )

                    title = obj.get(
                        "title",
                        ""
                    )

                    surah = obj.get(
                        "surah",
                        ""
                    )

                    ayah_range = obj.get(
                        "ayah_range",
                        ""
                    )

                    embedding_text = f"""
المصدر: {source_type}

العالم: {scholar}

العنوان: {title}

السورة: {surah}

الآيات: {ayah_range}

النص:
{base_text}
"""

                    embedding_text = normalize_arabic(
                        embedding_text
                    )

                    obj["embedding_text"] = embedding_text

                    records.append(obj)

                    texts.append(embedding_text)

                except Exception as e:

                    print(f"ERROR: {e}")

print(f"\nLoaded total records: {len(records)}")

# =====================================================
# BUILD BM25
# =====================================================

print("\nBuilding BM25 index...")

tokenized_corpus = [
    text.split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

print("\nGenerating embeddings...")

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

    print(
        f"Processed "
        f"{i + len(batch)} / {len(texts)}"
    )

embeddings = np.array(
    all_embeddings
).astype("float32")

print("\nEmbeddings generated.")

# =====================================================
# BUILD FAISS INDEX
# =====================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index ready.")

# =====================================================
# HYBRID SEARCH
# =====================================================

def hybrid_search(query):

    query = normalize_arabic(query)

    # =========================================
    # BM25 SEARCH
    # =========================================

    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:20]

    # =========================================
    # SEMANTIC SEARCH
    # =========================================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    semantic_scores, semantic_indices = (
        index.search(
            query_embedding,
            20
        )
    )

    # =========================================
    # MERGE
    # =========================================

    combined_scores = {}

    for idx in bm25_top_indices:

        combined_scores[idx] = (
            combined_scores.get(idx, 0)
            + (bm25_scores[idx] * 0.7)
        )

    for i, idx in enumerate(
        semantic_indices[0]
    ):

        combined_scores[idx] = (
            combined_scores.get(idx, 0)
            + (semantic_scores[0][i] * 0.3)
        )

    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:TOP_K]

# =====================================================
# SEARCH LOOP
# =====================================================

print("\n" + "=" * 70)
print("Unified Islamic Retrieval Engine Ready")
print("=" * 70)

while True:

    print("\n" + "=" * 70)

    query = input(
        "\nEnter your query (or 'exit'): "
    ).strip()

    if query.lower() == "exit":
        break

    results = hybrid_search(query)

    print("\nTOP RESULTS:\n")

    for rank, (idx, score) in enumerate(
        results,
        start=1
    ):

        result = records[idx]

        print("=" * 90)

        print(f"\nRank: {rank}")

        print(
            f"Combined Score: "
            f"{score:.4f}"
        )

        source_type = result.get(
            "source_type",
            "unknown"
        )

        scholar = result.get(
            "scholar",
            ""
        )

        metadata = (
            f"Source: {source_type} | "
            f"Scholar: {scholar}"
        )

        print(format_arabic(metadata))

        print("\nTEXT:\n")

        text = result.get(
            "clean_text",
            ""
        )[:2500]

        print(format_arabic(text))

        print("\n")

print("\nSystem exited.")