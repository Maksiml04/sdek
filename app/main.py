import os
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from app.models import ChatRequest, ChatResponse
from app.graph import build_graph
from app.rag import init_vectorstore
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CdekStart RAG Agent")
graph = build_graph()
sessions = {}


@app.on_event("startup")
def startup():
    init_vectorstore()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    state = sessions.get(req.session_id, {
        "messages": [],
        "country": None,
        "needs_clarification": False,
        "retrieved_context": ""
    })

    state["messages"].append(HumanMessage(content=req.message))

    result = await graph.ainvoke(state)
    sessions[req.session_id] = result

    last_msg = result["messages"][-1]
    return {"response": last_msg.content}