import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from app.models import ChatRequest, ChatResponse
from app.graph import build_graph
from app.rag import init_vectorstore
from app.memory import get_history, add_message, create_session_id, clear_history
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CdekStart RAG Agent")
graph = build_graph()

# Mount static files if needed
if os.path.exists("app/static"):
    app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def startup():
    init_vectorstore()


@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    """Serve the chat interface HTML page"""
    return FileResponse("app/templates/chat.html")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Если session_id не передан, создаем новый
    if not req.session_id:
        session_id = create_session_id()
    else:
        session_id = req.session_id
    
    # Получаем историю из хранилища
    history = get_history(session_id)
    
    # Формируем state с историей диалога
    state = {
        "messages": history.copy(),  # Копируем, чтобы не мутировать глобальное состояние до обработки
        "country": None,
        "needs_clarification": False,
        "retrieved_context": "",
        "is_general_query": False
    }

    # Добавляем новое сообщение пользователя
    user_msg = HumanMessage(content=req.message)
    add_message(session_id, user_msg)
    state["messages"].append(user_msg)

    # Запускаем граф
    result = await graph.ainvoke(state)
    
    # Сохраняем обновленную историю (с ответом бота) в хранилище
    # Берем все сообщения из результата и сохраняем
    for msg in result["messages"]:
        if msg not in history:
            add_message(session_id, msg)
    
    last_msg = result["messages"][-1]
    return {"response": last_msg.content, "session_id": session_id}


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Очищает историю диалога для указанной сессии"""
    clear_history(session_id)
    return {"message": "История очищена", "session_id": session_id}


@app.get("/chat/session/new")
async def create_new_session():
    """Создает новую сессию и возвращает её ID"""
    return {"session_id": create_session_id()}