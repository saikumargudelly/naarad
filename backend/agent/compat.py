"""Compatibility layer for LangChain and other dependencies.

This module provides compatibility functions and classes to handle differences
between different versions of LangChain and other dependencies.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Version compatibility checks
def get_langchain_version() -> str:
    """Get the installed LangChain version."""
    try:
        import langchain
        return langchain.__version__
    except ImportError:
        return "unknown"

def is_langchain_v1() -> bool:
    """Check if using LangChain v1.x."""
    version = get_langchain_version()
    return version.startswith("1.")

def is_langchain_v2() -> bool:
    """Check if using LangChain v2.x."""
    version = get_langchain_version()
    return version.startswith("2.")

def is_langchain_v3() -> bool:
    """Check if using LangChain v3.x."""
    version = get_langchain_version()
    return version.startswith("3.")

# LangChain imports with version compatibility
def get_langchain_imports():
    """Get LangChain imports based on version."""
    try:
        if is_langchain_v1():
            # LangChain v1.x imports
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain.agents.agent import AgentFinish, AgentAction
            from langchain.tools import BaseTool
            from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain.chains import LLMChain
            from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManager
            from langchain.schema.runnable import RunnableSequence
            from langchain_core.language_models import BaseChatModel
            from langchain_core.tools import BaseTool as LangChainBaseTool
            from langchain_core.messages import HumanMessage as CoreHumanMessage, AIMessage as CoreAIMessage, SystemMessage as CoreSystemMessage, BaseMessage as CoreBaseMessage
            from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate, MessagesPlaceholder as CoreMessagesPlaceholder
            from langchain_core.runnables import RunnablePassthrough, RunnableSequence as CoreRunnableSequence
            from langchain_core.callbacks.manager import CallbackManagerForChainRun as CoreCallbackManagerForChainRun
            from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            
            return {
                'AgentExecutor': AgentExecutor,
                'create_react_agent': create_react_agent,
                'AgentFinish': AgentFinish,
                'AgentAction': AgentAction,
                'BaseTool': BaseTool,
                'BaseMessage': BaseMessage,
                'HumanMessage': HumanMessage,
                'AIMessage': AIMessage,
                'SystemMessage': SystemMessage,
                'ChatPromptTemplate': ChatPromptTemplate,
                'MessagesPlaceholder': MessagesPlaceholder,
                'LLMChain': LLMChain,
                'CallbackManagerForChainRun': CallbackManagerForChainRun,
                'AsyncCallbackManager': AsyncCallbackManager,
                'RunnableSequence': RunnableSequence,
                'BaseChatModel': BaseChatModel,
                'LangChainBaseTool': LangChainBaseTool,
                'CoreHumanMessage': CoreHumanMessage,
                'CoreAIMessage': CoreAIMessage,
                'CoreSystemMessage': CoreSystemMessage,
                'CoreBaseMessage': CoreBaseMessage,
                'CoreChatPromptTemplate': CoreChatPromptTemplate,
                'CoreMessagesPlaceholder': CoreMessagesPlaceholder,
                'RunnablePassthrough': RunnablePassthrough,
                'CoreRunnableSequence': CoreRunnableSequence,
                'CoreCallbackManagerForChainRun': CoreCallbackManagerForChainRun,
                'StreamingStdOutCallbackHandler': StreamingStdOutCallbackHandler,
            }
        else:
            # LangChain v2.x/v3.x imports (current)
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain.agents.agent import AgentFinish, AgentAction
            from langchain_core.tools import BaseTool
            from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.chains import LLMChain
            from langchain_core.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManager
            from langchain_core.runnables import RunnableSequence
            from langchain_core.language_models import BaseChatModel
            from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            
            return {
                'AgentExecutor': AgentExecutor,
                'create_react_agent': create_react_agent,
                'AgentFinish': AgentFinish,
                'AgentAction': AgentAction,
                'BaseTool': BaseTool,
                'BaseMessage': BaseMessage,
                'HumanMessage': HumanMessage,
                'AIMessage': AIMessage,
                'SystemMessage': SystemMessage,
                'ChatPromptTemplate': ChatPromptTemplate,
                'MessagesPlaceholder': MessagesPlaceholder,
                'LLMChain': LLMChain,
                'CallbackManagerForChainRun': CallbackManagerForChainRun,
                'AsyncCallbackManager': AsyncCallbackManager,
                'RunnableSequence': RunnableSequence,
                'BaseChatModel': BaseChatModel,
                'LangChainBaseTool': BaseTool,
                'CoreHumanMessage': HumanMessage,
                'CoreAIMessage': AIMessage,
                'CoreSystemMessage': SystemMessage,
                'CoreBaseMessage': BaseMessage,
                'CoreChatPromptTemplate': ChatPromptTemplate,
                'CoreMessagesPlaceholder': MessagesPlaceholder,
                'RunnablePassthrough': RunnablePassthrough,
                'CoreRunnableSequence': RunnableSequence,
                'CoreCallbackManagerForChainRun': CallbackManagerForChainRun,
                'StreamingStdOutCallbackHandler': StreamingStdOutCallbackHandler,
            }
    except ImportError as e:
        logger.error(f"Failed to import LangChain components: {e}")
        raise

# Pydantic compatibility
def get_pydantic_model_config() -> Dict[str, Any]:
    """Get Pydantic model configuration based on version."""
    try:
        import pydantic
        version = pydantic.__version__
        
        if version.startswith("1."):
            # Pydantic v1.x
            return {
                'ConfigDict': None,  # Not available in v1
                'field_validator': 'validator',
                'model_validator': 'root_validator',
            }
        else:
            # Pydantic v2.x
            from pydantic import ConfigDict
            return {
                'ConfigDict': ConfigDict,
                'field_validator': 'field_validator',
                'model_validator': 'model_validator',
            }
    except ImportError:
        logger.error("Pydantic not found")
        raise

# FastAPI compatibility
def get_fastapi_imports():
    """Get FastAPI imports with version compatibility."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        
        return {
            'FastAPI': FastAPI,
            'HTTPException': HTTPException,
            'Request': Request,
            'Depends': Depends,
            'CORSMiddleware': CORSMiddleware,
            'JSONResponse': JSONResponse,
        }
    except ImportError as e:
        logger.error(f"Failed to import FastAPI components: {e}")
        raise

# Rate limiting compatibility
def get_rate_limiter():
    """Get rate limiter with compatibility."""
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        return Limiter, get_remote_address
    except ImportError as e:
        logger.error(f"Failed to import slowapi: {e}")
        raise

# Utility functions for compatibility
def create_compatible_model_config(**kwargs) -> Dict[str, Any]:
    """Create a compatible model configuration."""
    pydantic_config = get_pydantic_model_config()
    
    if pydantic_config['ConfigDict']:
        # Pydantic v2.x
        return pydantic_config['ConfigDict'](**kwargs)
    else:
        # Pydantic v1.x - return as dict
        return kwargs

def get_compatible_field_validator():
    """Get the compatible field validator decorator."""
    pydantic_config = get_pydantic_model_config()
    
    if pydantic_config['ConfigDict']:
        # Pydantic v2.x
        from pydantic import field_validator
        return field_validator
    else:
        # Pydantic v1.x
        from pydantic import validator
        return validator

# Version info
def get_version_info() -> Dict[str, str]:
    """Get version information for all major dependencies."""
    versions = {}
    
    try:
        import fastapi
        versions['fastapi'] = fastapi.__version__
    except ImportError:
        versions['fastapi'] = 'not installed'
    
    try:
        import pydantic
        versions['pydantic'] = pydantic.__version__
    except ImportError:
        versions['pydantic'] = 'not installed'
    
    try:
        import langchain
        versions['langchain'] = langchain.__version__
    except ImportError:
        versions['langchain'] = 'not installed'
    
    try:
        import uvicorn
        versions['uvicorn'] = uvicorn.__version__
    except ImportError:
        versions['uvicorn'] = 'not installed'
    
    try:
        import slowapi
        # slowapi doesn't have __version__ attribute, so we'll use 'installed'
        versions['slowapi'] = 'installed'
    except ImportError:
        versions['slowapi'] = 'not installed'
    
    return versions
