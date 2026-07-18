"""Evaluation suite for the Islamic RAG pipeline.

This module defines an evaluator that scores retrieval quality,
hallucination resistance, and citation grounding for queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import retrieval_engine
from answer_generator import AnswerGenerator
from indexer import load_or_build_index

REPORT_PATH = Path("evaluation_report.json")
HALLUCINATION_TRIGGER = "لم أجد جواباً صريحاً"

CATEGORY_SOURCE_MAP = {
    "fatwa": {"fatwa"},
    "hadith": {"hadith"},
    "tafsir": {"tafsir"},
    "quran": {"quran"},
}


class RetrievalEngineWrapper:
    def __init__(
        self,
        records: list[dict],
        tokenized_texts: list[list[str]],
        bm25: Any,
        index: Any,
        reranker: Any,
    ) -> None:
        self.records = records
        self.tokenized_texts = tokenized_texts
        self.bm25 = bm25
        self.index = index
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        candidates = retrieval_engine.search(
            query,
            self.records,
            self.tokenized_texts,
            self.bm25,
            self.index,
            self.reranker,
            top_k,
        )
        return [self.records[idx] for idx, _ in candidates]


class IslamicRAGEvaluator:
    def __init__(
        self,
        retrieval_engine: Any,
        answer_generator: Any,
        top_k: int = 10,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.answer_generator = answer_generator
        self.top_k = top_k

    def evaluate_query(
        self,
        query: str,
        expected_category: str,
        is_hallucination_test: bool = False,
    ) -> dict:
        """Evaluate a single query and return scoring metrics."""
        retrieved_docs = self._retrieve_documents(query)
        response = self.answer_generator.generate_answer(query, retrieved_docs)

        relevance_score = self._compute_relevance_score(retrieved_docs, expected_category)
        citation_score = self._compute_citation_score(response, is_hallucination_test)
        hallucination_score = self._compute_hallucination_score(response, is_hallucination_test)
        verification_status = bool(response.get("verified", False))

        result = {
            "query": query,
            "expected_category": expected_category,
            "is_hallucination_test": is_hallucination_test,
            "retrieved_documents": retrieved_docs,
            "response": response,
            "scores": {
                "relevance_score": relevance_score,
                "citation_score": citation_score,
                "hallucination_score": hallucination_score,
                "verification_status": verification_status,
            },
            "overall_score": self._compute_overall_score(
                relevance_score, citation_score, hallucination_score, verification_status
            ),
        }

        return result

    def save_report(self, results: list[dict]) -> Path:
        """Save the evaluation results to evaluation_report.json."""
        summary = self._summarize_results(results)
        payload = {
            "overall_score": summary["overall_score"],
            "fatwa_score": summary["category_scores"].get("fatwa", 0.0),
            "hadith_score": summary["category_scores"].get("hadith", 0.0),
            "tafsir_score": summary["category_scores"].get("tafsir", 0.0),
            "quran_score": summary["category_scores"].get("quran", 0.0),
            "hallucination_score": summary["hallucination_score"],
            "failed_tests": summary["failed_tests"],
            "recommendations": summary["recommendations"],
            "results": results,
        }

        REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return REPORT_PATH

    def _retrieve_documents(self, query: str) -> list[dict]:
        if hasattr(self.retrieval_engine, "search"):
            # Prefer search(query, top_k)
            try:
                return self.retrieval_engine.search(query, top_k=self.top_k)
            except TypeError:
                return self.retrieval_engine.search(query)

        if callable(self.retrieval_engine):
            return self.retrieval_engine(query)

        raise TypeError("Retrieval engine must be callable or have a search method.")

    def _compute_relevance_score(self, retrieved_docs: list[dict], expected_category: str) -> float:
        if not retrieved_docs:
            return 0.0

        expected_sources = CATEGORY_SOURCE_MAP.get(expected_category, set())
        if not expected_sources:
            return 0.0

        matched = 0
        total = 0
        for doc in retrieved_docs:
            source_type = str(doc.get("source_type", "")).lower().strip()
            if not source_type:
                continue
            total += 1
            if source_type in expected_sources:
                matched += 1

        return float(matched) / total if total else 0.0

    def _compute_citation_score(self, response: dict, is_hallucination_test: bool) -> float:
        sources = response.get("sources") or []
        if is_hallucination_test:
            return 1.0
        return 1.0 if len(sources) > 0 else 0.0

    def _compute_hallucination_score(self, response: dict, is_hallucination_test: bool) -> float:
        answer = str(response.get("answer", ""))
        if is_hallucination_test:
            return 1.0 if HALLUCINATION_TRIGGER in answer else 0.0
        return 1.0

    def _compute_overall_score(
        self,
        relevance_score: float,
        citation_score: float,
        hallucination_score: float,
        verification_status: bool,
    ) -> float:
        verification_bonus = 1.0 if verification_status else 0.0
        return round((relevance_score + citation_score + hallucination_score + verification_bonus) / 4.0, 4)

    def _summarize_results(self, results: list[dict]) -> dict:
        category_scores: dict[str, list[float]] = {"fatwa": [], "hadith": [], "tafsir": [], "quran": []}
        hallucination_scores: list[float] = []
        failed_tests: list[dict] = []
        recommendations: list[str] = []

        for result in results:
            expected = result.get("expected_category")
            score = result.get("overall_score", 0.0)
            if expected in category_scores:
                category_scores[expected].append(score)

            if result.get("is_hallucination_test"):
                hallucination_scores.append(result.get("scores", {}).get("hallucination_score", 0.0))

            if score < 0.75:
                failed_tests.append({
                    "query": result.get("query"),
                    "expected_category": expected,
                    "overall_score": score,
                    "scores": result.get("scores", {}),
                })

        category_average = {
            key: round(sum(values) / len(values), 4) if values else 0.0
            for key, values in category_scores.items()
        }
        overall_score = round(sum(category_average.values()) / len(category_average), 4) if category_average else 0.0
        hallucination_score = round(sum(hallucination_scores) / len(hallucination_scores), 4) if hallucination_scores else 0.0

        if failed_tests:
            recommendations.append(
                "راجع الاسترجاع والتوثيق لتقليل الأخطاء في الاختبارات التي فشلت."
            )
        if hallucination_scores and hallucination_score < 1.0:
            recommendations.append(
                "قم بتحسين مقاومة الهلوسة، خاصة في أسئلة لا يمكن الإجابة عليها من المصادر."
            )

        return {
            "overall_score": overall_score,
            "category_scores": category_average,
            "hallucination_score": hallucination_score,
            "failed_tests": failed_tests,
            "recommendations": recommendations,
        }

    def run_all_tests(self) -> list[dict]:
        categories = {
            "fatwa": [
                "ما حكم الإسبال؟",
                "هل يجوز حلق اللحية؟",
                "ما حكم سماع الأغاني؟",
            ],
            "hadith": [
                "حديث عن الأمانة",
                "حديث عن الصدق",
                "حديث عن الصلاة",
            ],
            "tafsir": [
                "ما معنى الصراط المستقيم؟",
                "تفسير سورة العصر",
                "تفسير قوله تعالى: إياك نعبد وإياك نستعين",
            ],
            "quran": [
                "آية عن الصبر",
                "آية عن التوبة",
                "آية عن الوالدين",
            ],
            "hallucination": [
                "من فاز بكأس العالم 2038؟",
                "ما حكم قيادة السيارة على سطح القمر؟",
                "ما هو سعر الذهب اليوم؟",
            ],
        }

        results: list[dict] = []

        for category, queries in categories.items():
            for query in queries:
                is_hallucination = category == "hallucination"
                expected_category = "quran" if category == "quran" else category
                try:
                    result = self.evaluate_query(query, expected_category, is_hallucination)
                except Exception as exc:
                    result = {
                        "query": query,
                        "expected_category": expected_category,
                        "is_hallucination_test": is_hallucination,
                        "retrieved_documents": [],
                        "response": {
                            "query": query,
                            "answer": "تعذر توليد الإجابة بسبب خطأ في النموذج.",
                            "verified": False,
                            "sources": [],
                        },
                        "scores": {
                            "relevance_score": 0.0,
                            "citation_score": 0.0,
                            "hallucination_score": 0.0,
                            "verification_status": False,
                        },
                        "overall_score": 0.0,
                        "error": str(exc),
                    }

                passed = self._is_passed(result)
                status = "PASS" if passed else "FAIL"

                print("=" * 80)
                print(f"Query: {query}")
                print(f"Expected: {category}")
                print(f"Status: {status}")
                print(f"Scores: {result['scores']}")
                if not passed:
                    print("Diagnosis:")
                    print(f"  Final answer: {result['response'].get('answer')}")
                    print(f"  Sources: {result['response'].get('sources')}")
                    print(f"  Verification: {result['scores'].get('verification_status')}")
                    if result.get("error"):
                        print(f"  Error: {result['error']}")
                results.append(result)

        print("=" * 80)
        self.save_report(results)
        print(f"Report written to: {REPORT_PATH}")
        return results

    def _is_passed(self, result: dict) -> bool:
        if result.get("is_hallucination_test"):
            return result.get("scores", {}).get("hallucination_score") == 1.0
        return result.get("overall_score", 0.0) >= 0.75

def main() -> None:
    logger = retrieval_engine.setup_logger(False)
    retrieval_engine.ensure_directories()

    records = retrieval_engine.load_processed_records(logger)
    if not records:
        print("No processed records found. Run data_pipeline.py first.")
        return

    index, embeddings, tokenized_texts, bm25 = load_or_build_index(records, rebuild=False, logger=logger)
    reranker = retrieval_engine.load_reranker()

    retrieval_wrapper = RetrievalEngineWrapper(records, tokenized_texts, bm25, index, reranker)
    answer_generator = AnswerGenerator()
    evaluator = IslamicRAGEvaluator(retrieval_wrapper, answer_generator)
    evaluator.run_all_tests()


if __name__ == "__main__":
    main()
