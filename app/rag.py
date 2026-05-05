import os
import shutil
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.parser import CdekStartParser

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class RAGService:
    """Сервис для работы с векторным хранилищем и поиска контекста."""

    def __init__(self, db_path: str = DB_PATH, embedding_model: str = EMBEDDING_MODEL_NAME):
        self.db_path = db_path
        self.data_dir = os.getenv("DATA_DIR", "/app/data")
        self.embedding_model_name = embedding_model
        self._vectorstore: Optional[Chroma] = None
        self._embedding: Optional[FastEmbedEmbeddings] = None

    def get_embedding_model(self) -> FastEmbedEmbeddings:
        if self._embedding is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding = FastEmbedEmbeddings(model_name=self.embedding_model_name)
        return self._embedding

    def init_vectorstore(self) -> Chroma:
        if self._vectorstore:
            return self._vectorstore

        logger.info(f"Starting vectorstore initialization...")
        logger.info(f"DB Path: {self.db_path}, Data Dir: {self.data_dir}")

        embedding = self.get_embedding_model()

        # 1. Проверка: существует ли база данных?
        if os.path.exists(self.db_path):
            logger.info(f"Found existing database at {self.db_path}. Loading...")
            try:
                self._vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=embedding,
                    collection_name="cdekstart_knowledge"
                )
                count = self._vectorstore._collection.count()
                logger.info(f"Successfully loaded existing database. Documents count: {count}")

                # === ИСПРАВЛЕНИЕ: Если база пустая, удаляем её и идем на пересоздание ===
                if count == 0:
                    logger.warning("Database is EMPTY! Deleting and re-indexing from data files...")
                    shutil.rmtree(self.db_path)
                    self._vectorstore = None
                    # Продолжаем выполнение кода ниже, где сработает ветка создания новой базы

            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                self._vectorstore = None

        # 2. Если базы нет (или она была удалена как пустая) -> Создаем новую
        if self._vectorstore is None:
            logger.warning(f"Creating new vectorstore...")
            logger.warning(f"Checking if data directory exists: {os.path.exists(self.data_dir)}")

            if not os.path.exists(self.data_dir):
                logger.error(f"DATA DIRECTORY NOT FOUND: {self.data_dir}. Cannot index documents!")
                # Создаем пустое хранилище, чтобы приложение не упало
                self._vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=embedding,
                    collection_name="cdekstart_knowledge"
                )
                return self._vectorstore

            # Парсинг документов
            logger.info(f"Starting parsing of files in {self.data_dir}...")
            parser = CdekStartParser(data_dir=self.data_dir)
            docs = parser.parse_all()

            logger.info(f"Parsing complete. Found {len(docs)} raw documents.")

            if len(docs) == 0:
                logger.error("No documents were parsed! Check file extensions and permissions.")
                self._vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=embedding,
                    collection_name="cdekstart_knowledge"
                )
                return self._vectorstore

            # Сплиттинг
            logger.info("Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
            split_docs = splitter.split_documents(docs)
            logger.info(f"Splitting complete. Total chunks: {len(split_docs)}.")

            # Создание базы
            logger.info(f"Creating Chroma vectorstore with {len(split_docs)} chunks...")
            try:
                self._vectorstore = Chroma.from_documents(
                    split_docs,
                    embedding,
                    persist_directory=self.db_path,
                    collection_name="cdekstart_knowledge"
                )
                logger.info("Vectorstore created and saved successfully.")

                final_count = self._vectorstore._collection.count()
                logger.info(f"Final check: Database contains {final_count} documents.")

            except Exception as e:
                logger.error(f"Critical error during vectorstore creation: {e}", exc_info=True)
                raise e

        return self._vectorstore

    def retrieve_context(self, query: str, country: Optional[str] = None, k: int = 3) -> str:
        vs = self.init_vectorstore()

        filter_meta = {"country": country} if country else None

        # Поиск с фильтром
        results = vs.similarity_search(query, k=k, filter=filter_meta)

        # Если ничего не найдено, ищем без фильтра
        if not results:
            logger.info(f"No results with country filter, searching without filter")
            results = vs.similarity_search(query, k=k)

        logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
        return "\n".join([doc.page_content for doc in results])

    def retrieve_documents(self, query: str, country: Optional[str] = None, k: int = 3) -> List[Document]:
        vs = self.init_vectorstore()
        filter_meta = {"country": country} if country else None
        results = vs.similarity_search(query, k=k, filter=filter_meta)

        if not results:
            results = vs.similarity_search(query, k=k)
        return results

    def get_vectorstore(self):
        if self._vectorstore is None:
            self.init_vectorstore()
        return self._vectorstore


# Глобальный экземпляр
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# Функции для совместимости
def init_vectorstore() -> Chroma:
    return get_rag_service().init_vectorstore()


def retrieve_context(query: str, country: Optional[str] = None) -> str:
    return get_rag_service().retrieve_context(query, country)