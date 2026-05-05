"""
Модуль управления памятью диалогов.
Хранит историю сообщений для каждого session_id.
В продакшене рекомендуется заменить на Redis или базу данных.
"""
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import uuid

# Глобальное хранилище (в памяти)
# Структура: { "session_id": [HumanMessage(...), AIMessage(...), ...] }
conversation_store: Dict[str, List[BaseMessage]] = {}

def get_history(session_id: str) -> List[BaseMessage]:
    """Получает историю сообщений для сессии."""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]

def add_message(session_id: str, message: BaseMessage):
    """Добавляет сообщение в историю сессии."""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    conversation_store[session_id].append(message)

def clear_history(session_id: str):
    """Очищает историю для конкретной сессии."""
    if session_id in conversation_store:
        del conversation_store[session_id]

def create_session_id() -> str:
    """Генерирует новый уникальный ID сессии."""
    return str(uuid.uuid4())
