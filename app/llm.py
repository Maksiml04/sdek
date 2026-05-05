# app/llm.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    """
    Создаёт LLM-клиент для OpenAI-совместимых провайдеров:
    - OpenRouter (DeepSeek, Llama, Claude и др.)
    - OpenAI
    - Прямой DeepSeek API
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    model_name = os.getenv("MODEL_NAME", "deepseek/deepseek-chat")

    # Базовые параметры для всех провайдеров
    common_kwargs = {
        "model": model_name,
        "temperature": 0.0,  # детерминированные ответы для RAG
    }

    if provider == "openrouter":
        return ChatOpenAI(
            **common_kwargs,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
                "X-Title": os.getenv("APP_NAME", "CdekStart RAG Agent"),
            }
        )

    elif provider == "deepseek":
        # Прямой доступ к DeepSeek API (если есть ключ от deepseek.com)
        return ChatOpenAI(
            **common_kwargs,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
        )

    else:
        # Стандартный OpenAI (fallback)
        return ChatOpenAI(
            **common_kwargs,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )