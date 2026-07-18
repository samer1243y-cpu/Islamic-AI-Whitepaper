from __future__ import annotations

import json
import sys

from rag_agent import RAGAgent


def main() -> None:
    print("بدء تشغيل RAG التفاعلي...")
    try:
        agent = RAGAgent(debug=False)
    except Exception as exc:
        print(f"تعذر تهيئة النظام: {exc}")
        return

    while True:
        try:
            user_query = input("أدخل سؤالك الشرعي (أو 'خروج' للإنهاء): ").strip()
        except KeyboardInterrupt:
            print("\nتم إيقاف التشغيل.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"خروج", "exit", "quit"}:
            print("تم إنهاء الجلسة.")
            break

        try:
            print("\n" + "=" * 70)
            print("🔍 الاستعلام الأصلي")
            print(user_query)

            optimized_json = agent.query_optimizer.optimize_query(user_query)
            try:
                optimized_data = json.loads(optimized_json)
                optimized_query = optimized_data.get("optimized_query", "") or user_query
            except Exception:
                optimized_query = user_query

            print("\n🧠 الاستعلام المحسّن")
            print(optimized_query)

            result = agent.run_query(user_query)

            print("\n📝 الإجابة النهائية")
            print(result.get("answer", "لا توجد إجابة"))

            print("\n📚 المصادر المستخدمة")
            sources = result.get("sources", [])
            if sources:
                for idx, src in enumerate(sources, start=1):
                    label = src.get("scholar") or src.get("book") or src.get("source") or "مصدر غير معروف"
                    print(f"{idx}. {label}")
            else:
                print("لا توجد مصادر")

            print("\n" + "=" * 70)
        except Exception as exc:
            message = str(exc).lower()
            if "429" in message or "rate limit" in message:
                print("عذراً، حدثت مشكلة في الحدّ الأقصى للطلبات من خدمة الذكاء الاصطناعي. حاول مرة أخرى لاحقاً.")
            else:
                print(f"حدث خطأ غير متوقع: {exc}")


if __name__ == "__main__":
    main()
