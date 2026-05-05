# app/rag.py
import os
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.parser import CdekStartParser

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")  # Путь внутри контейнера
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class RAGService:
    """Сервис для работы с векторным хранилищем и поиска контекста."""
    
    def __init__(self, db_path: str = DB_PATH, embedding_model: str = EMBEDDING_MODEL_NAME):
        """Инициализация RAG сервиса.
        
        Args:
            db_path: Путь к базе данных Chroma.
            embedding_model: Название модели для эмбеддингов.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self._vectorstore: Optional[Chroma] = None
        self._embedding: Optional[FastEmbedEmbeddings] = None
    
    def get_embedding_model(self) -> FastEmbedEmbeddings:
        """Создаёт или возвращает кэшированную модель эмбеддингов.
        
        Returns:
            Модель эмбеддингов FastEmbed.
        """
        if self._embedding is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding = FastEmbedEmbeddings(
                model_name=self.embedding_model_name,
            )
        return self._embedding
    
    def init_vectorstore(self) -> Chroma:
        """Инициализирует векторное хранилище.
        
        Загружает существующее хранилище или создаёт новое из файлов данных.
        
        Returns:
            Инициализированное векторное хранилище Chroma.
        """
        if self._vectorstore:
            return self._vectorstore

        embedding = self.get_embedding_model()

        if os.path.exists(self.db_path):
            logger.info(f"Loading existing vectorstore from {self.db_path}")
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=embedding,
                collection_name="cdekstart_knowledge"
            )
        else:
            logger.info("Creating new vectorstore from data files using CdekStartParser")
            
            parser = CdekStartParser(data_dir=DATA_DIR)
            docs = parser.parse_all()

            logger.info(f"Parsed {len(docs)} documents")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            split_docs = splitter.split_documents(docs)

            logger.info(f"Indexing {len(split_docs)} chunks...")
            self._vectorstore = Chroma.from_documents(
                split_docs,
                embedding,
                persist_directory=self.db_path,
                collection_name="cdekstart_knowledge"
            )
            self._vectorstore.persist()
            logger.info("Vectorstore created and saved")

        return self._vectorstore
    
    def retrieve_context(self, query: str, country: Optional[str] = None, k: int = 3) -> str:
        """Ищет релевантный контекст по запросу.
        
        Args:
            query: Текст запроса пользователя.
            country: Опциональный фильтр по стране.
            k: Количество документов для поиска.
            
        Returns:
            Строка с найденным контекстом.
        """
        vs = self.init_vectorstore()
        
        # Пробуем поиск с фильтром по стране
        filter_meta = {"country": country} if country else {}
        results = vs.similarity_search(query, k=k, filter=filter_meta)
        
        # Если ничего не найдено, ищем без фильтра
        if not results:
            logger.info(f"No results with country filter, searching without filter")
            results = vs.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
        
        return "\n".join([doc.page_content for doc in results])
    
    def retrieve_documents(self, query: str, country: Optional[str] = None, k: int = 3) -> List[Document]:
        """Ищет релевантные документы по запросу.
        
        Args:
            query: Текст запроса пользователя.
            country: Опциональный фильтр по стране.
            k: Количество документов для поиска.
            
        Returns:
            Список найденных документов LangChain.
        """
        vs = self.init_vectorstore()
        
        filter_meta = {"country": country} if country else {}
        results = vs.similarity_search(query, k=k, filter=filter_meta)
        
        if not results:
            results = vs.similarity_search(query, k=k)
        
        return results


# Глобальный экземпляр для обратной совместимости
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Возвращает глобальный экземпляр RAG сервиса.
    
    Returns:
        Экземпляр RAGService.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# Функции для обратной совместимости
def init_vectorstore() -> Chroma:
    """Инициализирует векторное хранилище (для обратной совместимости).
    
    Returns:
        Инициализированное векторное хранилище Chroma.
    """
    return get_rag_service().init_vectorstore()


def retrieve_context(query: str, country: Optional[str] = None) -> str:
    """Ищет контекст (для обратной совместимости).
    
    Args:
        query: Текст запроса.
        country: Опциональный фильтр по стране.
        
    Returns:
        Строка с найденным контекстом.
    """
    return get_rag_service().retrieve_context(query, country)