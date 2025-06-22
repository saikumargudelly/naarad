from typing import Dict, List, Any, Optional, Union, Awaitable, Callable, TypeVar
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer, select, update, insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import os
import asyncio
import logging
from functools import wraps
from llm.config import settings

# Set up logger
logger = logging.getLogger(__name__)

# Import Supabase client
try:
    from .supabase_client import supabase_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    supabase_client = None

# Type variable for generic return type
T = TypeVar('T')

def sync_to_async(fn: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Convert a sync function to async."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    return wrapper

# Initialize SQLAlchemy
Base = declarative_base()

class ConversationMemory(Base):
    __tablename__ = 'conversation_memories'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    conversation_id = Column(String, index=True)
    messages = Column(JSON)
    metadata_ = Column('metadata', JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    agent_states = Column(JSON)  # Store agent-specific states

class MemoryManager:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite+aiosqlite:///naarad_memory.db")
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            future=True
        )
        self.async_session = async_sessionmaker(
            self.engine, 
            expire_on_commit=False,
            class_=AsyncSession
        )
        # Don't initialize DB here, will be done on first use
        self._initialized = False
        
        # Initialize Supabase if available
        self.supabase_enabled = SUPABASE_AVAILABLE and supabase_client and supabase_client.is_enabled()
        if self.supabase_enabled:
            logger.info("Supabase integration enabled")
        else:
            logger.info("Supabase integration disabled - using SQLite only")
        
    async def ensure_initialized(self):
        """Ensure the database is initialized."""
        if not self._initialized:
            await self._init_db()
            self._initialized = True
    
    async def _init_db(self):
        """Initialize the database and create tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_conversation(self, conversation_id: str, user_id: str = "default") -> Optional[Dict]:
        """Retrieve a conversation by ID with Supabase fallback."""
        # Try Supabase first if enabled
        if self.supabase_enabled:
            try:
                supabase_result = await supabase_client.get_conversation(conversation_id, user_id)
                if supabase_result:
                    logger.debug(f"Retrieved conversation {conversation_id} from Supabase")
                    return supabase_result
            except Exception as e:
                logger.warning(f"Supabase retrieval failed, falling back to SQLite: {e}")
        
        # Fallback to SQLite
        await self.ensure_initialized()
        async with self.async_session() as session:
            result = await session.execute(
                select(ConversationMemory)
                .where(
                    (ConversationMemory.conversation_id == conversation_id) &
                    (ConversationMemory.user_id == user_id)
                )
            )
            memory = result.scalars().first()
            
            if memory:
                return {
                    'id': memory.id,
                    'user_id': memory.user_id,
                    'conversation_id': memory.conversation_id,
                    'messages': memory.messages or [],
                    'metadata': memory.metadata_ or {},
                    'agent_states': memory.agent_states or {}
                }
        return None
    
    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        user_id: str = "default",
        metadata: Optional[Dict] = None,
        agent_states: Optional[Dict] = None
    ) -> Dict:
        """Save or update a conversation with dual storage (Supabase + SQLite)."""
        # Save to Supabase if enabled
        supabase_result = None
        if self.supabase_enabled:
            try:
                supabase_result = await supabase_client.save_conversation(
                    conversation_id, messages, user_id, metadata, agent_states
                )
                if supabase_result:
                    logger.debug(f"Saved conversation {conversation_id} to Supabase")
            except Exception as e:
                logger.warning(f"Supabase save failed: {e}")
        
        # Always save to SQLite as backup
        await self.ensure_initialized()
        memory_id = f"{user_id}_{conversation_id}"
        
        async with self.async_session() as session:
            # Check if conversation exists
            result = await session.execute(
                select(ConversationMemory)
                .where(ConversationMemory.id == memory_id)
            )
            memory = result.scalars().first()
            
            if memory:
                # Update existing conversation
                memory.messages = messages
                memory.metadata_ = metadata or memory.metadata_
                memory.agent_states = agent_states or memory.agent_states
                memory.updated_at = datetime.utcnow()
            else:
                # Create new conversation
                memory = ConversationMemory(
                    id=memory_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=messages,
                    metadata_=metadata or {},
                    agent_states=agent_states or {}
                )
                session.add(memory)
            
            await session.commit()
            await session.refresh(memory)
            
            sqlite_result = {
                'id': memory.id,
                'user_id': memory.user_id,
                'conversation_id': memory.conversation_id,
                'messages': memory.messages or [],
                'metadata': memory.metadata_ or {},
                'agent_states': memory.agent_states or {}
            }
        
        # Return Supabase result if available, otherwise SQLite result
        return supabase_result if supabase_result else sqlite_result
    
    async def delete_conversation(self, conversation_id: str, user_id: str = "default") -> bool:
        """Delete a conversation from both Supabase and SQLite."""
        success = True
        
        # Delete from Supabase if enabled
        if self.supabase_enabled:
            try:
                supabase_success = await supabase_client.delete_conversation(conversation_id, user_id)
                if supabase_success:
                    logger.debug(f"Deleted conversation {conversation_id} from Supabase")
                else:
                    logger.warning(f"Failed to delete conversation {conversation_id} from Supabase")
                    success = False
            except Exception as e:
                logger.warning(f"Supabase deletion failed: {e}")
                success = False
        
        # Delete from SQLite
        await self.ensure_initialized()
        memory_id = f"{user_id}_{conversation_id}"
        
        async with self.async_session() as session:
            result = await session.execute(
                select(ConversationMemory)
                .where(ConversationMemory.id == memory_id)
            )
            memory = result.scalars().first()
            
            if memory:
                await session.delete(memory)
                await session.commit()
                logger.debug(f"Deleted conversation {conversation_id} from SQLite")
            else:
                logger.warning(f"Conversation {conversation_id} not found in SQLite")
                success = False
        
        return success
    
    async def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations for a user, preferring Supabase if available."""
        if self.supabase_enabled:
            try:
                supabase_conversations = await supabase_client.list_conversations(user_id, limit)
                if supabase_conversations:
                    logger.debug(f"Retrieved {len(supabase_conversations)} conversations from Supabase")
                    return supabase_conversations
            except Exception as e:
                logger.warning(f"Supabase list failed, falling back to SQLite: {e}")
        
        # Fallback to SQLite
        await self.ensure_initialized()
        async with self.async_session() as session:
            result = await session.execute(
                select(ConversationMemory)
                .where(ConversationMemory.user_id == user_id)
                .order_by(ConversationMemory.updated_at.desc())
                .limit(limit)
            )
            memories = result.scalars().all()
            
            return [
                {
                    'id': memory.id,
                    'conversation_id': memory.conversation_id,
                    'metadata': memory.metadata_ or {},
                    'created_at': memory.created_at.isoformat() if memory.created_at else None,
                    'updated_at': memory.updated_at.isoformat() if memory.updated_at else None
                }
                for memory in memories
            ]
    
    def update_agent_state(
        self,
        conversation_id: str,
        agent_name: str,
        state: Dict,
        user_id: str = "default"
    ) -> Dict:
        """Update the state of a specific agent in a conversation."""
        with self.Session() as session:
            memory = session.query(ConversationMemory).filter_by(
                conversation_id=conversation_id,
                user_id=user_id
            ).first()
            
            if not memory:
                memory = ConversationMemory(
                    id=f"{user_id}_{conversation_id}",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=[],
                    metadata_={},
                    agent_states={}
                )
                session.add(memory)
            
            agent_states = memory.agent_states or {}
            agent_states[agent_name] = state
            memory.agent_states = agent_states
            
            session.commit()
            
            return agent_states
    
    def get_agent_state(
        self,
        conversation_id: str,
        agent_name: str,
        user_id: str = "default"
    ) -> Optional[Dict]:
        """Get the state of a specific agent in a conversation."""
        memory = self.get_conversation(conversation_id, user_id)
        if memory and 'agent_states' in memory:
            return memory['agent_states'].get(agent_name)
        return None

# Singleton instance
memory_manager = MemoryManager()

# For backward compatibility with sync code
sync_memory_manager = MemoryManager()
sync_memory_manager.engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///naarad_memory.db"))
sync_memory_manager.Session = sessionmaker(bind=sync_memory_manager.engine)

# Patch the async methods to work in sync context
sync_memory_manager.get_conversation = sync_to_async(sync_memory_manager.get_conversation)
sync_memory_manager.save_conversation = sync_to_async(sync_memory_manager.save_conversation)
