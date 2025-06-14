from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
from pydantic import BaseModel

from routers import chat as chat_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Naarad AI Assistant",
    description="A curious, cat-like AI companion that assists users via chat",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Include routers
app.include_router(chat_router.router, prefix="/api", tags=["chat"])

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    content: str
    images: list[str] = []
    conversation_id: str = ""

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Naarad AI Assistant is running! 😺",
        "docs": "/docs",
        "openapi_schema": "/openapi.json"
    }

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        # Process the message with Naarad's agent
        response = {
            "message": "I'm Naarad, your cat-like AI assistant! 😺 I'm still learning, but I'm here to help with your questions and tasks.",
            "sources": []
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("APP_ENV") == "development"
    )
