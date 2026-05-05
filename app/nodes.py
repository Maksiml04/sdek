import os
import re
from typing import Annotated, TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.memory import get_memory_store
from app.rag import get_rag_service

# === КОНСТАНТЫ ===
GENERAL_TOPICS = [
    "льгот", "benefit", "дедлайн", "deadline", "срок",
    "жиль", "housing", "проезд", "travel", "страхов", "insurance",
    "виз", "visa", "документ", "document"
]

COUNTRY_PATTERNS = {
    "germany": r"(германи|deutschland|berlin|берлин)",
    "france": r"(франци|france|paris|париж)",
    # Добавьте другие страны по необходимости
}

# === ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ ===
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-chat")

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY не найден! Проверьте .env")

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
)

rag_service = get_rag_service()
memory_store = get_memory_store()


# === ТИПЫ ДАННЫХ ===
class State(TypedDict):
    messages: list
    session_id: str
    country: Optional[str]
    topic: Optional[str]


# === УЗЛЫ ГРАФА ===

def check_country(state: State) -> State:
    """Определяет, нужна ли страна для ответа, или вопрос общий."""
    messages = state["messages"]
    last_msg = messages[-1].content.lower() if messages else ""

    # 1. Проверяем общие темы (не требуют страну)
    is_general_query = any(topic in last_msg for topic in GENERAL_TOPICS)

    if is_general_query:
        # Если вопрос общий, но пользователь упомянул страну, можно уточнить
        # Но пока просто пропускаем поиск страны для общих вопросов
        return {"country": None, "topic": "general"}

    # 2. Пытаемся найти страну в сообщении
    detected_country = None
    for country, pattern in COUNTRY_PATTERNS.items():
        if re.search(pattern, last_msg):
            detected_country = country
            break

    return {"country": detected_country, "topic": "specific" if detected_country else "unknown"}


def clarify_node(state: State) -> State:
    """Запрашивает уточнение, если тема не ясна."""
    question = AIMessage(content="Уточните, пожалуйста, о какой стране или аспекте программы вы хотите узнать?")
    return {"messages": [question]}


def retrieve_node(state: State) -> State:
    query = state["messages"][-1].content
    country = state.get("country")

    # Метод уже возвращает готовую строку с контекстом
    context_text = rag_service.retrieve_context(query, country=country)

    if not context_text or context_text.strip() == "":
        context_text = "Контекст не найден."

    system_msg = SystemMessage(content=f"Используй следующий контекст для ответа:\n{context_text}")
    new_messages = state["messages"][:-1] + [system_msg, state["messages"][-1]]

    return {"messages": new_messages}


def generate_node(state: State) -> State:
    """Генерирует ответ с помощью LLM."""
    messages = state["messages"]

    # Добавляем системный промпт, если его нет в начале
    if not isinstance(messages[0], SystemMessage):
        system_prompt = (
            "Ты полезный ассистент СДЭК.Start. Отвечай вежливо и по существу. "
            "Если информации в контексте недостаточно, скажи об этом честно."
        )
        messages = [SystemMessage(content=system_prompt)] + messages

    try:
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Ошибка LLM: {str(e)}"
        print(f"❌ {error_msg}")
        return {"messages": [AIMessage(content="Извините, произошла ошибка при генерации ответа.")]}