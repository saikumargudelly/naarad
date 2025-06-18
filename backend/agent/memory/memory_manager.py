from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import os
from llm.config import settings

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
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///naarad_memory.db")
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create tables."""
        Base.metadata.create_all(self.engine)
    
    def get_conversation(self, conversation_id: str, user_id: str = "default") -> Optional[Dict]:
        """Retrieve a conversation by ID."""
        with self.Session() as session:
            memory = session.query(ConversationMemory).filter_by(
                conversation_id=conversation_id,
                user_id=user_id
            ).first()
            
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
    
    def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        user_id: str = "default",
        metadata: Optional[Dict] = None,
        agent_states: Optional[Dict] = None
    ) -> Dict:
        """Save or update a conversation."""
        with self.Session() as session:
            memory = session.query(ConversationMemory).filter_by(
                conversation_id=conversation_id,
                user_id=user_id
            ).first()
            
            if memory:
                memory.messages = messages
                memory.metadata_ = metadata or memory.metadata_
                memory.agent_states = agent_states or memory.agent_states
            else:
                memory = ConversationMemory(
                    id=f"{user_id}_{conversation_id}",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=messages,
                    metadata_=metadata or {},
                    agent_states=agent_states or {}
                )
                session.add(memory)
            
            session.commit()
            
            return {
                'id': memory.id,
                'user_id': memory.user_id,
                'conversation_id': memory.conversation_id,
                'messages': memory.messages,
                'metadata': memory.metadata_,
                'agent_states': memory.agent_states
            }
    
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
