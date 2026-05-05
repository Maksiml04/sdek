import os
from app.rag import get_rag_service

# Убедимся, что переменные окружения подхватываются (если есть .env)
from dotenv import load_dotenv

load_dotenv()


def main():
    print("🚀 Инициализация RAG сервиса...")
    try:
        rag_service = get_rag_service()
        vs = rag_service.get_vectorstore()
        if vs:
            print(f"📊 В базе документов: {vs._collection.count()}")
        else:
            print("❌ Векторное хранилище не загружено!")

        rag_service = get_rag_service()

        # Принудительно инициализируем хранилище (загружаем модель и базу)
        print("📂 Загрузка векторного хранилища...")
        rag_service.init_vectorstore()

        test_queries = [
            "benefits стажировки",
            "для кого открыта программа",
            "deadlines подачи заявок",
            "СДЭК условия"
        ]

        for query in test_queries:
            print(f"\n{'=' * 40}")
            print(f"🔍 ЗАПРОС: {query}")
            print(f"{'=' * 40}")

            # Получаем сырые документы (k=5 для большего контекста)
            docs = rag_service.retrieve_documents(query, k=5)

            if not docs:
                print("❌ Документы не найдены!")
            else:
                print(f"✅ Найдено документов: {len(docs)}")
                for i, doc in enumerate(docs, 1):
                    print(f"\n--- Документ #{i} (Score: {doc.metadata.get('score', 'N/A')}) ---")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content: {doc.page_content[:1000]}...")  # Первые 200 символов

            # Проверка метода retrieve_context (который возвращает строку)
            context_str = rag_service.retrieve_context(query, k=3)
            print(f"\n{query}\n📝 Сформированный контекст (для LLM):\n{context_str[:1000]}...")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()