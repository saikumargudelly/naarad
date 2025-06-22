"""Supabase client for database operations."""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel, Field
import json

logger = logging.getLogger(__name__)

class ConversationRecord(BaseModel):
    """Model for conversation records in Supabase."""
    id: str = Field(description="Unique identifier for the conversation")
    user_id: str = Field(description="User identifier")
    conversation_id: str = Field(description="Conversation identifier")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    agent_states: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific states")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class SupabaseClient:
    """Supabase client for database operations."""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """Initialize Supabase client.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not provided. Supabase integration will be disabled.")
            self.client = None
            self.enabled = False
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                self.enabled = True
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
                self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if Supabase client is enabled and working."""
        return self.enabled and self.client is not None
    
    async def get_conversation(self, conversation_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve a conversation from Supabase.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            
        Returns:
            Conversation data or None if not found
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return None
        
        try:
            response = self.client.table("conversation_memories").select("*").eq(
                "conversation_id", conversation_id
            ).eq("user_id", user_id).execute()
            
            if response.data:
                record = response.data[0]
                return {
                    'id': record['id'],
                    'user_id': record['user_id'],
                    'conversation_id': record['conversation_id'],
                    'messages': record.get('messages', []),
                    'metadata': record.get('metadata', {}),
                    'agent_states': record.get('agent_states', {}),
                    'created_at': record.get('created_at'),
                    'updated_at': record.get('updated_at')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving conversation from Supabase: {e}")
            return None
    
    async def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        agent_states: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Save or update a conversation in Supabase.
        
        Args:
            conversation_id: Unique conversation identifier
            messages: List of conversation messages
            user_id: User identifier
            metadata: Additional metadata
            agent_states: Agent-specific states
            
        Returns:
            Saved conversation data or None if failed
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return None
        
        try:
            now = datetime.utcnow().isoformat()
            record_id = f"{user_id}_{conversation_id}"
            
            # Check if conversation exists
            existing = await self.get_conversation(conversation_id, user_id)
            
            if existing:
                # Update existing conversation
                response = self.client.table("conversation_memories").update({
                    'messages': messages,
                    'metadata': metadata or existing.get('metadata', {}),
                    'agent_states': agent_states or existing.get('agent_states', {}),
                    'updated_at': now
                }).eq("id", record_id).execute()
            else:
                # Create new conversation
                response = self.client.table("conversation_memories").insert({
                    'id': record_id,
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                    'messages': messages,
                    'metadata': metadata or {},
                    'agent_states': agent_states or {},
                    'created_at': now,
                    'updated_at': now
                }).execute()
            
            if response.data:
                record = response.data[0] if isinstance(response.data, list) else response.data
                return {
                    'id': record['id'],
                    'user_id': record['user_id'],
                    'conversation_id': record['conversation_id'],
                    'messages': record.get('messages', []),
                    'metadata': record.get('metadata', {}),
                    'agent_states': record.get('agent_states', {}),
                    'created_at': record.get('created_at'),
                    'updated_at': record.get('updated_at')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error saving conversation to Supabase: {e}")
            return None
    
    async def delete_conversation(self, conversation_id: str, user_id: str = "default") -> bool:
        """Delete a conversation from Supabase.
        
        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return False
        
        try:
            record_id = f"{user_id}_{conversation_id}"
            response = self.client.table("conversation_memories").delete().eq("id", record_id).execute()
            return bool(response.data)
            
        except Exception as e:
            logger.error(f"Error deleting conversation from Supabase: {e}")
            return False
    
    async def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return []
        
        try:
            response = self.client.table("conversation_memories").select(
                "id, conversation_id, metadata, created_at, updated_at"
            ).eq("user_id", user_id).order("updated_at", desc=True).limit(limit).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error listing conversations from Supabase: {e}")
            return []
    
    async def update_agent_state(
        self,
        conversation_id: str,
        agent_name: str,
        state: Dict[str, Any],
        user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Update agent state for a conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            agent_name: Name of the agent
            state: Agent state data
            user_id: User identifier
            
        Returns:
            Updated agent states or None if failed
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return None
        
        try:
            existing = await self.get_conversation(conversation_id, user_id)
            if not existing:
                logger.warning(f"Conversation {conversation_id} not found")
                return None
            
            agent_states = existing.get('agent_states', {})
            agent_states[agent_name] = state
            
            result = await self.save_conversation(
                conversation_id=conversation_id,
                messages=existing.get('messages', []),
                user_id=user_id,
                metadata=existing.get('metadata', {}),
                agent_states=agent_states
            )
            
            return result.get('agent_states') if result else None
            
        except Exception as e:
            logger.error(f"Error updating agent state in Supabase: {e}")
            return None
    
    async def get_agent_state(
        self,
        conversation_id: str,
        agent_name: str,
        user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get agent state for a conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            agent_name: Name of the agent
            user_id: User identifier
            
        Returns:
            Agent state or None if not found
        """
        if not self.is_enabled():
            logger.warning("Supabase client not enabled")
            return None
        
        try:
            conversation = await self.get_conversation(conversation_id, user_id)
            if conversation and 'agent_states' in conversation:
                return conversation['agent_states'].get(agent_name)
            return None
            
        except Exception as e:
            logger.error(f"Error getting agent state from Supabase: {e}")
            return None

# Global Supabase client instance
supabase_client = SupabaseClient()
