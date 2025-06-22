"""Main FastAPI application for Naarad AI Assistant."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from contextlib import asynccontextmanager

from config.config import settings
from config.logging_config import setup_logging
from routers import chat, websocket, voice, analytics, personalization

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Naarad AI Assistant...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Naarad AI Assistant...")

# Create FastAPI app
app = FastAPI(
    title="Naarad AI Assistant",
    description="Advanced AI Assistant with Multi-Agent Architecture",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
app.include_router(voice.router, prefix="/api/v1", tags=["voice"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(personalization.router, prefix="/api/v1", tags=["personalization"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Naarad AI Assistant",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Multi-Agent Architecture",
            "Real-time WebSocket Chat",
            "Voice Processing",
            "Analytics & Insights",
            "Personalization",
            "Image Analysis"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "naarad-ai-assistant",
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT
    }

@app.get("/api/v1/health")
async def api_health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "api_version": "v1",
        "endpoints": {
            "chat": "/api/v1/chat",
            "websocket": "/api/v1/ws",
            "voice": "/api/v1/voice",
            "analytics": "/api/v1/analytics",
            "personalization": "/api/v1/personalization"
        },
        "features": {
            "real_time_streaming": True,
            "voice_processing": True,
            "analytics": True,
            "personalization": True,
            "image_analysis": True
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
