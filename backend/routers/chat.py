import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.rag import build_vector_db, stream_rag_response

router = APIRouter(prefix="/chat", tags=["chat"])

_vector_db = None


def _get_vector_db():
    # Built lazily (not at import time) so the rest of the backend keeps
    # working even when OPENAI_API_KEY isn't configured.
    global _vector_db
    if _vector_db is None:
        _vector_db = build_vector_db()
    return _vector_db


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4o-mini"


@router.post("")
def chat(payload: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    if not payload.messages or payload.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user.")

    try:
        vector_db = _get_vector_db()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize the chatbot's knowledge base: {exc}")

    if vector_db is None:
        raise HTTPException(status_code=500, detail="No reference documents found for the chatbot.")

    messages = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in payload.messages
    ]

    llm = ChatOpenAI(model=payload.model, streaming=True)

    return StreamingResponse(
        stream_rag_response(vector_db, llm, messages),
        media_type="text/plain",
    )
