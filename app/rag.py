# app/rag.py
import os
import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.parser import CdekStartParser

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

_vectorstore = None


def get_embedding_model():
    """Создаёт лёгкую модель эмбеддингов через fastembed"""
    return FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )


def init_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore:
        return _vectorstore

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding = get_embedding_model()

    if os.path.exists(DB_PATH):
        logger.info(f"Loading existing vectorstore from {DB_PATH}")
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding,
            collection_name="cdekstart_knowledge"
        )
    else:
        logger.info("Creating new vectorstore from data files using CdekStartParser")
        
        parser = CdekStartParser(data_dir="data/")
        docs = parser.parse_all()

        logger.info(f"Parsed {len(docs)} documents")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        split_docs = splitter.split_documents(docs)

        logger.info(f"Indexing {len(split_docs)} chunks...")
        _vectorstore = Chroma.from_documents(
            split_docs,
            embedding,
            persist_directory=DB_PATH,
            collection_name="cdekstart_knowledge"
        )
        _vectorstore.persist()
        logger.info("Vectorstore created and saved")

    return _vectorstore


def retrieve_context(query: str, country: str | None) -> str:
    vs = init_vectorstore()
    filter_meta = {"country": country} if country else {}
    results = vs.similarity_search(query, k=3, filter=filter_meta)
    if not results:
        results = vs.similarity_search(query, k=3)
    logger.info(f"Retrieved {len(results)} documents")
    return "\n".join([doc.page_content for doc in results])