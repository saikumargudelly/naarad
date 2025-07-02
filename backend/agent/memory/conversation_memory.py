from typing import List, Dict, Any
from datetime import datetime

class ConversationMemory:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, Any]] = []  # Each: {'role': 'user'/'assistant', 'content': str, 'timestamp': datetime}
        self.topics: List[str] = []
        self.intents: List[str] = []
        self.entities: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_message(self, role: str, content: str):
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat()
        })

    def add_topic(self, topic: str):
        if topic and topic not in self.topics:
            self.topics.append(topic)

    def add_intent(self, intent: str):
        if intent and intent not in self.intents:
            self.intents.append(intent)

    def add_entities(self, entities: List[Dict[str, Any]]):
        self.entities.extend(entities) 