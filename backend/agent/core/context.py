from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
import uuid

class ConversationContext(BaseModel):
    """Represents the context for a conversation."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    user_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the user"
    )
    conversation_id: str = Field(
        default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the conversation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the conversation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was last updated"
    )
    
    def update_timestamp(self) -> 'ConversationContext':
        """Update the last updated timestamp.
        
        Returns:
            ConversationContext: The updated conversation context
        """
        self.updated_at = datetime.utcnow()
        return self
