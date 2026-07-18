from answer_generator import AnswerGenerator

if __name__ == "__main__":
    # هذا السكربت يطلب السؤال من المستخدم بالعربية ويخرج عند كتابة exit.
    # تأكد من أن .env يحتوي على GROQ_API_KEY صالح.
    docs = [
        {
            "source_type": "tafsir",
            "source": "تفسير ابن كثير",
            "book": "تفسير ابن كثير",
            "surah": "1",
            "ayah": "1",
            "text": "قال ابن كثير إن الصراط المستقيم طريق الله المبين.",
            "score": 9.0,
        }
    ]

    generator = AnswerGenerator()

    while True:
        query = input("أدخل السؤال أو اكتب exit للخروج: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("تم الخروج.")
            break

        response = generator.generate_answer(query, docs)
        print("\nالسؤال:", response["query"])
        print("التحقق:", "ناجح" if response["verified"] else "فشل")
        print("الإجابة:", response["answer"])
        print("المصادر:", response["sources"])
        print("-" * 60)
