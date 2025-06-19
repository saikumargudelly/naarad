import logging
import sys
from pathlib import Path
from typing import Optional

from config.config import settings

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=settings.LOG_FORMAT,
        handlers=handlers
    )
    
    # Set log levels for specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").handlers = logging.getLogger("uvicorn").handlers
