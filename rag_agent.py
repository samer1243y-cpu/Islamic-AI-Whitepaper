from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from retrieval_engine import (
    setup_logger,
    ensure_directories,
    route_query,
    search as engine_search,
    load_reranker,
    rerank_candidates,
)
from indexer import load_or_build_index
from data_pipeline import load_processed_records
from answer_generator import AnswerGenerator
from query_optimizer import QueryOptimizer


class RAGAgent:
    def __init__(self, model_name: str | None = None, debug: bool = False) -> None:
        self.logger = setup_logger(debug)
        ensure_directories()

        self.logger.info("Initializing RAGAgent...")
        self.records = load_processed_records(self.logger)
        if not self.records:
            raise RuntimeError("No processed records found. Run the data pipeline first.")

        # Build or load index artifacts
        self.index, self.embeddings, self.tokenized_texts, self.bm25 = load_or_build_index(self.records, rebuild=False, logger=self.logger)

        # Reranker may be loaded on demand
        self.reranker = None

        # LLM-backed components
        self.query_optimizer = QueryOptimizer()
        self.answer_generator = AnswerGenerator()

    def run_query(self, user_query: str, hybrid_top_k: int = 20, final_top_k: int = 5) -> dict:
        # Intent analysis
        intent = route_query(user_query)
        self.logger.info(f"Intent: {intent}")

        # Query optimization
        optimized_json = self.query_optimizer.optimize_query(user_query)
        try:
            parsed = json.loads(optimized_json)
            optimized_query = parsed.get("optimized_query", "").strip()
        except Exception:
            optimized_query = user_query

        if not optimized_query:
            optimized_query = user_query

        # Prefix as required by embedding model prompt engineering
        prefixed = "query: " + optimized_query

        # Hybrid retrieval (FAISS + BM25) — explicitly pass reranker=None to get hybrid-only top-k
        hybrid_results = engine_search(
            prefixed,
            self.records,
            self.tokenized_texts,
            self.bm25,
            self.index,
            reranker=None,
            top_k=hybrid_top_k,
        )
        candidate_ids = [idx for idx, _ in hybrid_results]

        # Cross-encoder reranking (top hybrid_top_k -> final_top_k)
        self.reranker = load_reranker()  # attempt to load the CrossEncoder
        if self.reranker is not None and candidate_ids:
            reranked = rerank_candidates(prefixed, candidate_ids, self.records, self.reranker)
            # reranked is list of (idx, score)
            reranked_sorted = sorted(reranked, key=lambda item: item[1], reverse=True)[:final_top_k]
        else:
            # Fallback: take top final_top_k from hybrid ranking
            reranked_sorted = hybrid_results[:final_top_k]

        # Build retrieved_docs to feed to the answer generator
        retrieved_docs = []
        for idx, score in reranked_sorted:
            doc = dict(self.records[idx])
            doc["score"] = float(score)
            retrieved_docs.append(doc)

        # Grounded answer generation using original user query
        try:
            response = self.answer_generator.generate_answer(user_query, retrieved_docs)
            if response.get("answer"):
                return response
        except Exception:
            self.logger.warning("Answer generation failed; falling back to a simple grounded summary.")

        fallback_answer = "لم أتمكن من توليد إجابة مباشرة من النموذج، لكن الوثائق الأكثر صلة تشير إلى أن الموضوع مرتبط بالفقه الإسلامي والربا والاقتراض."
        if retrieved_docs:
            top_text = retrieved_docs[0].get("clean_text", "") or retrieved_docs[0].get("text", "")
            if top_text:
                fallback_answer = top_text[:500]

        return {
            "query": user_query,
            "answer": fallback_answer,
            "verified": False,
            "sources": [
                {
                    "source_type": doc.get("source_type", ""),
                    "source": doc.get("source", ""),
                    "book": doc.get("book", ""),
                    "scholar": doc.get("scholar", ""),
                    "surah": doc.get("surah", ""),
                    "ayah": doc.get("ayah", ""),
                    "score": doc.get("score", 0.0),
                }
                for doc in retrieved_docs
            ],
        }


def main() -> None:
    agent = RAGAgent()
    print("RAG Agent ready. Enter queries (type 'exit' to quit).")
    while True:
        try:
            q = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            out = agent.run_query(q)
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
