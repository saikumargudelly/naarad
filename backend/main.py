"""Main FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from config.config import settings
from config.logging_setup import setup_logging
from routers import chat as chat_router

# Import compatibility layer
from agent.compat import get_version_info

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
    
    # Log version information
    versions = get_version_info()
    logger.info("Dependency versions:")
    for dep, version in versions.items():
        logger.info(f"  {dep}: {version}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Naarad AI Assistant...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Naarad AI Assistant - A multi-agent AI system",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.security.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("slowapi not installed, rate limiting disabled")

# Include routers
app.include_router(
    chat_router.router,
    prefix=settings.API_V1_STR,
    tags=["chat"]
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Naarad AI Assistant",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "naarad-ai",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        log_level=settings.logging.level.lower()
    )
