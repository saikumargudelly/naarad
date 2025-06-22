-- Supabase Database Setup Script for Naarad AI Assistant
-- Run this script in your Supabase SQL editor to create the necessary tables

-- Enable Row Level Security (RLS)
ALTER TABLE IF EXISTS conversation_memories ENABLE ROW LEVEL SECURITY;

-- Create the conversation_memories table
CREATE TABLE IF NOT EXISTS conversation_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    messages JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    agent_states JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversation_memories_user_id ON conversation_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_memories_conversation_id ON conversation_memories(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_memories_updated_at ON conversation_memories(updated_at DESC);

-- Create a unique constraint on user_id and conversation_id combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_memories_user_conversation 
ON conversation_memories(user_id, conversation_id);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_conversation_memories_updated_at 
    BEFORE UPDATE ON conversation_memories 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create RLS policies for security
-- Policy for users to read their own conversations
CREATE POLICY "Users can read their own conversations" ON conversation_memories
    FOR SELECT USING (auth.uid()::text = user_id);

-- Policy for users to insert their own conversations
CREATE POLICY "Users can insert their own conversations" ON conversation_memories
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Policy for users to update their own conversations
CREATE POLICY "Users can update their own conversations" ON conversation_memories
    FOR UPDATE USING (auth.uid()::text = user_id);

-- Policy for users to delete their own conversations
CREATE POLICY "Users can delete their own conversations" ON conversation_memories
    FOR DELETE USING (auth.uid()::text = user_id);

-- Create a view for conversation summaries (useful for listing conversations)
CREATE OR REPLACE VIEW conversation_summaries AS
SELECT 
    id,
    user_id,
    conversation_id,
    metadata,
    created_at,
    updated_at,
    jsonb_array_length(messages) as message_count
FROM conversation_memories;

-- Grant necessary permissions
GRANT ALL ON conversation_memories TO authenticated;
GRANT ALL ON conversation_summaries TO authenticated;

-- Create a function to get conversation statistics
CREATE OR REPLACE FUNCTION get_conversation_stats(user_id_param TEXT)
RETURNS TABLE(
    total_conversations BIGINT,
    total_messages BIGINT,
    oldest_conversation TIMESTAMP WITH TIME ZONE,
    newest_conversation TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_conversations,
        COALESCE(SUM(jsonb_array_length(messages)), 0) as total_messages,
        MIN(created_at) as oldest_conversation,
        MAX(updated_at) as newest_conversation
    FROM conversation_memories 
    WHERE user_id = user_id_param;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_conversation_stats(TEXT) TO authenticated;

-- Insert some sample data for testing (optional)
-- INSERT INTO conversation_memories (id, user_id, conversation_id, messages, metadata) VALUES
-- ('test_user_test_conv_1', 'test_user', 'test_conv_1', 
--  '[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]'::jsonb,
--  '{"title": "Test Conversation", "tags": ["test"]}'::jsonb);

-- Create a function to clean up old conversations (optional)
CREATE OR REPLACE FUNCTION cleanup_old_conversations(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM conversation_memories 
    WHERE updated_at < NOW() - INTERVAL '1 day' * days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the cleanup function
GRANT EXECUTE ON FUNCTION cleanup_old_conversations(INTEGER) TO authenticated;

-- Create a function to search conversations by content
CREATE OR REPLACE FUNCTION search_conversations(
    user_id_param TEXT,
    search_term TEXT,
    limit_param INTEGER DEFAULT 10
)
RETURNS TABLE(
    id TEXT,
    conversation_id TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    relevance_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cm.id,
        cm.conversation_id,
        cm.metadata,
        cm.created_at,
        cm.updated_at,
        GREATEST(
            CASE WHEN cm.metadata::text ILIKE '%' || search_term || '%' THEN 1.0 ELSE 0.0 END,
            CASE WHEN cm.messages::text ILIKE '%' || search_term || '%' THEN 0.8 ELSE 0.0 END
        ) as relevance_score
    FROM conversation_memories cm
    WHERE cm.user_id = user_id_param
    AND (
        cm.metadata::text ILIKE '%' || search_term || '%' OR
        cm.messages::text ILIKE '%' || search_term || '%'
    )
    ORDER BY relevance_score DESC, cm.updated_at DESC
    LIMIT limit_param;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the search function
GRANT EXECUTE ON FUNCTION search_conversations(TEXT, TEXT, INTEGER) TO authenticated;

-- Add comments for documentation
COMMENT ON TABLE conversation_memories IS 'Stores conversation history and metadata for Naarad AI Assistant';
COMMENT ON COLUMN conversation_memories.id IS 'Unique identifier combining user_id and conversation_id';
COMMENT ON COLUMN conversation_memories.user_id IS 'User identifier for conversation ownership';
COMMENT ON COLUMN conversation_memories.conversation_id IS 'Unique conversation identifier';
COMMENT ON COLUMN conversation_memories.messages IS 'JSON array of conversation messages';
COMMENT ON COLUMN conversation_memories.metadata IS 'Additional metadata for the conversation';
COMMENT ON COLUMN conversation_memories.agent_states IS 'Agent-specific state information';
COMMENT ON COLUMN conversation_memories.created_at IS 'Timestamp when conversation was created';
COMMENT ON COLUMN conversation_memories.updated_at IS 'Timestamp when conversation was last updated'; 