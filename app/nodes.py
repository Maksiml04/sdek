import re
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage
from app.llm import get_llm
from app.rag import retrieve_context


# Темы, которые не требуют указания страны
GENERAL_TOPICS = [
    "benefits", "deadlines", "general", "программа", "участие", 
    "отбор", "язык", "дедлайн", "дата", "срок", "жильё", "проезд", 
    "страховка", "сертификат", "выгода"
]


def check_country(state: dict) -> dict:
    """Проверяет, нужна ли страна для ответа на запрос.
    
    Если запрос касается общих тем (benefits, deadlines, программа),
    страна не требуется. Если запрос специфичен для локации — требуется.
    """
    last_msg = state["messages"][-1].content if state["messages"] else ""
    country = state.get("country")
    lower_msg = last_msg.lower()
    
    # Проверяем, является ли запрос общим (не требует страну)
    is_general_query = any(topic in lower_msg for topic in GENERAL_TOPICS)
    
    # Пытаемся извлечь страну из сообщения, если она не указана в state
    if not country:
        if re.search(r"(германи|germany|берлин|berlin)", lower_msg):
            country = "germany"
        elif re.search(r"(франци|france|париж|paris)", lower_msg):
            country = "france"
    
    # Если запрос общий — страна не нужна, идём сразу на retrieve
    if is_general_query:
        return {"country": None, "needs_clarification": False, "is_general_query": True}
    
    # Если запрос специфичный и страна не найдена — нужна уточнение
    if not country:
        return {"country": None, "needs_clarification": True, "is_general_query": False}
    
    return {"country": country, "needs_clarification": False, "is_general_query": False}


def clarify_node(state: dict) -> dict:
    return {"messages": [AIMessage(content="Пожалуйста, уточните страну стажировки (Германия или Франция)?")]}


def retrieve_node(state: dict) -> dict:
    query = state["messages"][-1].content
    context = retrieve_context(query, state["country"])
    return {"retrieved_context": context}


def generate_node(state: dict) -> dict:
    llm = get_llm()
    system_prompt = (
        "Ты консультант программы стажировки CdekStart.\n"
        "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
        "Если в контексте нет ответа, напиши: 'В базе знаний нет информации по этому вопросу.'\n"
        "Никогда не выдумывай цифры, сроки или правила.\n\n"
        f"Контекст:\n{state['retrieved_context']}"
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}