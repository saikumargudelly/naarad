# Supabase Integration Guide for Naarad AI Assistant

## 📋 Overview

This document provides a comprehensive guide for setting up and using Supabase integration with the Naarad AI Assistant. The integration provides a robust, scalable database solution for storing conversation history and agent states.

## 🎯 What's Included

- **Dual Storage System**: Supabase as primary, SQLite as fallback
- **Automatic Failover**: Seamless fallback to SQLite if Supabase is unavailable
- **Row Level Security**: Built-in security policies
- **Real-time Capabilities**: Future-ready for real-time features
- **Scalable Architecture**: PostgreSQL-based with automatic scaling

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install supabase>=2.0.0
```

### 2. Set Up Supabase Project

1. Go to [Supabase](https://supabase.com) and create a new project
2. Wait for the project to be ready
3. Go to Settings > API and copy your credentials:
   - **Project URL** (SUPABASE_URL)
   - **anon public** key (SUPABASE_KEY)

### 3. Configure Environment

Create or edit your `.env` file:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
```

### 4. Set Up Database Tables

1. Go to the SQL Editor in your Supabase dashboard
2. Copy and paste the contents of `scripts/supabase_setup.sql`
3. Run the SQL script

### 5. Test the Integration

```bash
cd backend
python scripts/supabase_setup.py
```

## 🏗️ Architecture

### Dual Storage System

The system uses a hybrid approach:

```
┌─────────────────┐    ┌─────────────────┐
│   Supabase      │    │     SQLite      │
│   (Primary)     │    │   (Fallback)    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┼───┐
                                 │   │
                    ┌─────────────▼───▼─────────────┐
                    │      Memory Manager           │
                    │  (Automatic Failover Logic)   │
                    └───────────────────────────────┘
```

### Storage Priority

1. **Primary**: Supabase (PostgreSQL)
2. **Fallback**: SQLite (local file)
3. **Automatic**: Seamless switching between them

## 📊 Database Schema

### conversation_memories Table

```sql
CREATE TABLE conversation_memories (
    id TEXT PRIMARY KEY,                    -- user_id_conversation_id
    user_id TEXT NOT NULL,                  -- User identifier
    conversation_id TEXT NOT NULL,          -- Conversation identifier
    messages JSONB DEFAULT '[]'::jsonb,     -- Conversation messages
    metadata JSONB DEFAULT '{}'::jsonb,     -- Additional metadata
    agent_states JSONB DEFAULT '{}'::jsonb, -- Agent-specific states
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes for Performance

- `idx_conversation_memories_user_id` - Fast user lookups
- `idx_conversation_memories_conversation_id` - Fast conversation lookups
- `idx_conversation_memories_updated_at` - Fast sorting by date
- `idx_conversation_memories_user_conversation` - Unique constraint

## 🔒 Security Features

### Row Level Security (RLS)

The system includes comprehensive RLS policies:

```sql
-- Users can only access their own conversations
CREATE POLICY "Users can read their own conversations" 
ON conversation_memories FOR SELECT 
USING (auth.uid()::text = user_id);

-- Users can only insert their own conversations
CREATE POLICY "Users can insert their own conversations" 
ON conversation_memories FOR INSERT 
WITH CHECK (auth.uid()::text = user_id);

-- Users can only update their own conversations
CREATE POLICY "Users can update their own conversations" 
ON conversation_memories FOR UPDATE 
USING (auth.uid()::text = user_id);

-- Users can only delete their own conversations
CREATE POLICY "Users can delete their own conversations" 
ON conversation_memories FOR DELETE 
USING (auth.uid()::text = user_id);
```

### API Key Security

- Uses the **anon public** key (safe for client-side use)
- RLS policies enforce data isolation
- No sensitive operations exposed

## 🛠️ Usage Examples

### Basic Usage

The integration is transparent to your application code:

```python
from agent.memory.memory_manager import memory_manager

# Save a conversation (automatically uses Supabase if available)
await memory_manager.save_conversation(
    conversation_id="conv_123",
    messages=[{"role": "user", "content": "Hello"}],
    user_id="user_456"
)

# Retrieve a conversation
conversation = await memory_manager.get_conversation("conv_123", "user_456")

# List conversations
conversations = await memory_manager.list_conversations("user_456")

# Delete a conversation
success = await memory_manager.delete_conversation("conv_123", "user_456")
```

### Advanced Usage

```python
# Save with metadata and agent states
await memory_manager.save_conversation(
    conversation_id="conv_123",
    messages=messages,
    user_id="user_456",
    metadata={
        "title": "AI Discussion",
        "tags": ["ai", "discussion"],
        "model_used": "gpt-4"
    },
    agent_states={
        "researcher": {"search_count": 5},
        "analyst": {"analysis_depth": "deep"}
    }
)

# Update agent state
await memory_manager.update_agent_state(
    conversation_id="conv_123",
    agent_name="researcher",
    state={"search_count": 10},
    user_id="user_456"
)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Your Supabase project URL | None |
| `SUPABASE_KEY` | Your Supabase anon key | None |
| `DATABASE_URL` | SQLite fallback URL | `sqlite:///naarad_memory.db` |

### Configuration Classes

```python
from config.database_config import DatabaseSettings

# Database settings include Supabase configuration
database_settings = DatabaseSettings()
print(database_settings.SUPABASE_URL)
print(database_settings.SUPABASE_KEY)
```

## 📈 Performance Considerations

### Connection Pooling

Supabase automatically handles connection pooling, but you can configure:

```python
# In your Supabase client configuration
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
```

### Caching Strategy

- **Read Operations**: Supabase first, SQLite fallback
- **Write Operations**: Dual write (Supabase + SQLite)
- **Delete Operations**: Both databases

### Monitoring

Enable logging to monitor Supabase operations:

```python
import logging
logging.getLogger('agent.memory.supabase_client').setLevel(logging.DEBUG)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "Supabase client not enabled"

**Cause**: Missing or invalid credentials
**Solution**: Check your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`

#### 2. "relation 'conversation_memories' does not exist"

**Cause**: Database tables not created
**Solution**: Run the SQL setup script in Supabase dashboard

#### 3. "Failed to initialize Supabase client"

**Cause**: Network issues or invalid URL
**Solution**: Check your internet connection and URL format

#### 4. "RLS policy violation"

**Cause**: User not authenticated or policy mismatch
**Solution**: Ensure proper user authentication or use service role key

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Connection

Use the setup script to test:

```bash
python scripts/supabase_setup.py
```

## 🔄 Migration from SQLite

### Automatic Migration

The system automatically handles migration:

1. **New Data**: Goes to Supabase (if available)
2. **Existing Data**: Stays in SQLite
3. **Dual Write**: New data written to both systems

### Manual Migration

To migrate existing SQLite data to Supabase:

```python
import asyncio
from agent.memory.memory_manager import memory_manager

async def migrate_data():
    # Get all conversations from SQLite
    conversations = await memory_manager.list_conversations("all_users")
    
    for conv in conversations:
        # Save to Supabase
        await memory_manager.save_conversation(
            conv['conversation_id'],
            conv['messages'],
            conv['user_id'],
            conv['metadata']
        )

# Run migration
asyncio.run(migrate_data())
```

## 🔮 Future Enhancements

### Planned Features

1. **Real-time Updates**: Live conversation updates
2. **Vector Search**: Semantic search capabilities
3. **Analytics**: Conversation analytics and insights
4. **Backup/Restore**: Automated backup solutions
5. **Multi-tenancy**: Enhanced multi-tenant support

### Extensibility

The architecture supports easy extension:

```python
# Custom storage backends
class CustomStorageBackend:
    async def save_conversation(self, ...):
        # Custom implementation
        pass

# Integrate with memory manager
memory_manager.add_backend(CustomStorageBackend())
```

## 📚 API Reference

### SupabaseClient

```python
class SupabaseClient:
    def __init__(self, supabase_url: str, supabase_key: str)
    def is_enabled(self) -> bool
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict]
    async def save_conversation(self, conversation_id: str, messages: List, user_id: str, ...) -> Optional[Dict]
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool
    async def list_conversations(self, user_id: str, limit: int = 50) -> List[Dict]
    async def update_agent_state(self, conversation_id: str, agent_name: str, state: Dict, user_id: str) -> Optional[Dict]
    async def get_agent_state(self, conversation_id: str, agent_name: str, user_id: str) -> Optional[Dict]
```

### MemoryManager

```python
class MemoryManager:
    def __init__(self, db_url: str = None)
    async def get_conversation(self, conversation_id: str, user_id: str = "default") -> Optional[Dict]
    async def save_conversation(self, conversation_id: str, messages: List, user_id: str = "default", ...) -> Dict
    async def delete_conversation(self, conversation_id: str, user_id: str = "default") -> bool
    async def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[Dict]
    def update_agent_state(self, conversation_id: str, agent_name: str, state: Dict, user_id: str = "default") -> Dict
    def get_agent_state(self, conversation_id: str, agent_name: str, user_id: str = "default") -> Optional[Dict]
```

## 🤝 Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Supabase project
4. Run tests: `pytest tests/`
5. Run setup script: `python scripts/supabase_setup.py`

### Testing

```bash
# Run all tests
pytest tests/

# Run Supabase-specific tests
pytest tests/ -k "supabase"

# Run with coverage
pytest tests/ --cov=agent.memory
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Test with the setup script
4. Create an issue in the repository

## 📄 License

This integration is part of the Naarad AI Assistant project and follows the same license terms. 