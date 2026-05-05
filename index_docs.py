from app.rag import get_rag_service
import os


def main():
    # Инициализация сервиса
    rag_service = get_rag_service()

    print("🚀 Запуск индексации...")

    # Метод init_vectorstore сам проверит наличие базы.
    # Если базы нет -> сам распарсит data_dir (из переменных окружения) и создаст базу.
    # Если база есть -> просто загрузит её.
    try:
        vs = rag_service.init_vectorstore()
        count = vs._collection.count()
        print(f"✅ Векторное хранилище готово. Всего чанков в базе: {count}")
    except Exception as e:
        print(f"❌ Ошибка при инициализации: {e}")
        return

    print("✨ Индексация завершена!")


if __name__ == "__main__":
    main()

# import os
# print(f"Текущая директория: {os.getcwd()}")
# print(f"Путь к data: {os.path.abspath('data')}")
# print(f"Существует ли папка? {os.path.exists('data')}")
# if os.path.exists('data'):
#     print(f"Файлы внутри: {os.listdir('data')}")

# debug_parser.py
# from app.parser import CdekStartParser
#
# parser = CdekStartParser(data_dir="data")
# docs = parser.parse_all()
# print(f"Найдено документов: {len(docs)}")
# if docs:
#     print(f"Пример контента первого документа: {docs[0].page_content[:100]}...")
#     print(f"Метаданные: {docs[0].metadata}")
# else:
#     print("Документы не найдены! Проверьте содержимое папки data.")