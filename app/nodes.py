import re
from langchain_core.messages import AIMessage, SystemMessage
from app.llm import get_llm
from app.rag import retrieve_context


def check_country(state: dict) -> dict:
    last_msg = state["messages"][-1].content if state["messages"] else ""
    country = state.get("country")

    if not country:
        lower = last_msg.lower()
        if re.search(r"(германи|germany|берлин|berlin)", lower):
            country = "germany"
        elif re.search(r"(франци|france|париж|paris)", lower):
            country = "france"

    if not country:
        return {"country": None, "needs_clarification": True}
    return {"country": country, "needs_clarification": False}


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