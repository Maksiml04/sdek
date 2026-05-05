# app/memory.py
from typing import Dict, List
import uuid

class MemoryStore:
    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}

    def get_session(self, session_id: str) -> List[dict]:
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, message: dict):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Глобальный экземпляр (Singleton)
_memory_store_instance = None

def get_memory_store() -> MemoryStore:
    global _memory_store_instance
    if _memory_store_instance is None:
        _memory_store_instance = MemoryStore()
    return _memory_store_instance

# === Функции-обёртки для удобства импорта в main.py ===

def create_session_id() -> str:
    """Генерирует уникальный ID сессии."""
    return str(uuid.uuid4())

def get_history(session_id: str) -> List[dict]:
    """Получает историю сообщений для сессии."""
    store = get_memory_store()
    return store.get_session(session_id)

def add_message(session_id: str, message: dict):
    """Добавляет сообщение в историю сессии."""
    store = get_memory_store()
    store.add_message(session_id, message)

def clear_history(session_id: str):
    """Очищает историю сессии."""
    store = get_memory_store()
    store.clear_session(session_id)