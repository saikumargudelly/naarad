"""
Test suite for Supabase integration.

This test suite verifies that the Supabase integration works correctly
with the memory manager and provides proper fallback to SQLite.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

class TestSupabaseIntegration:
    """Test cases for Supabase integration."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        with patch('agent.memory.supabase_client.create_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_supabase_response(self):
        """Mock Supabase response."""
        mock_response = MagicMock()
        mock_response.data = [{
            'id': 'test_user_test_conv',
            'user_id': 'test_user',
            'conversation_id': 'test_conv',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'metadata': {'title': 'Test Conversation'},
            'agent_states': {'researcher': {'search_count': 5}},
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T00:00:00Z'
        }]
        return mock_response
    
    @pytest.mark.asyncio
    async def test_supabase_client_initialization_with_credentials(self, mock_supabase_client):
        """Test Supabase client initialization with valid credentials."""
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            assert client.is_enabled() is True
            assert client.client is not None
    
    @pytest.mark.asyncio
    async def test_supabase_client_initialization_without_credentials(self):
        """Test Supabase client initialization without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            assert client.is_enabled() is False
            assert client.client is None
    
    @pytest.mark.asyncio
    async def test_supabase_get_conversation_success(self, mock_supabase_client, mock_supabase_response):
        """Test successful conversation retrieval from Supabase."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_supabase_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            result = await client.get_conversation('test_conv', 'test_user')
            
            assert result is not None
            assert result['conversation_id'] == 'test_conv'
            assert result['user_id'] == 'test_user'
            assert len(result['messages']) == 1
            assert result['messages'][0]['content'] == 'Hello'
    
    @pytest.mark.asyncio
    async def test_supabase_get_conversation_not_found(self, mock_supabase_client):
        """Test conversation retrieval when not found in Supabase."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            result = await client.get_conversation('nonexistent_conv', 'test_user')
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_supabase_save_conversation_new(self, mock_supabase_client, mock_supabase_response):
        """Test saving a new conversation to Supabase."""
        # Mock get_conversation to return None (new conversation)
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        # Mock insert response
        mock_insert_response = MagicMock()
        mock_insert_response.data = [{
            'id': 'test_user_test_conv',
            'user_id': 'test_user',
            'conversation_id': 'test_conv',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'metadata': {},
            'agent_states': {},
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T00:00:00Z'
        }]
        
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            messages = [{'role': 'user', 'content': 'Hello'}]
            
            result = await client.save_conversation('test_conv', messages, 'test_user')
            
            assert result is not None
            assert result['conversation_id'] == 'test_conv'
            assert result['user_id'] == 'test_user'
    
    @pytest.mark.asyncio
    async def test_memory_manager_supabase_fallback(self):
        """Test memory manager fallback to SQLite when Supabase is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            from agent.memory.memory_manager import memory_manager
            
            # Test that it falls back to SQLite
            test_messages = [{'role': 'user', 'content': 'Test message'}]
            
            # Save conversation (should use SQLite)
            result = await memory_manager.save_conversation(
                'test_fallback_conv',
                test_messages,
                'test_user'
            )
            
            assert result is not None
            assert result['conversation_id'] == 'test_fallback_conv'
            assert result['user_id'] == 'test_user'
            
            # Retrieve conversation (should use SQLite)
            retrieved = await memory_manager.get_conversation('test_fallback_conv', 'test_user')
            
            assert retrieved is not None
            assert retrieved['messages'] == test_messages
    
    @pytest.mark.asyncio
    async def test_memory_manager_dual_storage(self, mock_supabase_client, mock_supabase_response):
        """Test memory manager with both Supabase and SQLite storage."""
        # Mock Supabase to be available
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_supabase_response
        
        # Mock insert response
        mock_insert_response = MagicMock()
        mock_insert_response.data = [{
            'id': 'test_user_test_dual_conv',
            'user_id': 'test_user',
            'conversation_id': 'test_dual_conv',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'metadata': {},
            'agent_states': {},
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T00:00:00Z'
        }]
        
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = mock_insert_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.memory_manager import memory_manager
            
            test_messages = [{'role': 'user', 'content': 'Hello'}]
            
            # Save conversation (should save to both Supabase and SQLite)
            result = await memory_manager.save_conversation(
                'test_dual_conv',
                test_messages,
                'test_user'
            )
            
            assert result is not None
            assert result['conversation_id'] == 'test_dual_conv'
    
    @pytest.mark.asyncio
    async def test_supabase_error_handling(self, mock_supabase_client):
        """Test error handling when Supabase operations fail."""
        # Mock Supabase to raise an exception
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("Supabase error")
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            result = await client.get_conversation('test_conv', 'test_user')
            
            # Should return None on error
            assert result is None
    
    @pytest.mark.asyncio
    async def test_supabase_list_conversations(self, mock_supabase_client):
        """Test listing conversations from Supabase."""
        mock_response = MagicMock()
        mock_response.data = [
            {
                'id': 'test_user_conv1',
                'conversation_id': 'conv1',
                'metadata': {'title': 'Conversation 1'},
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z'
            },
            {
                'id': 'test_user_conv2',
                'conversation_id': 'conv2',
                'metadata': {'title': 'Conversation 2'},
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-01T00:00:00Z'
            }
        ]
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            result = await client.list_conversations('test_user', limit=10)
            
            assert len(result) == 2
            assert result[0]['conversation_id'] == 'conv1'
            assert result[1]['conversation_id'] == 'conv2'
    
    @pytest.mark.asyncio
    async def test_supabase_delete_conversation(self, mock_supabase_client):
        """Test deleting a conversation from Supabase."""
        mock_response = MagicMock()
        mock_response.data = [{'id': 'test_user_test_conv'}]  # Deleted record
        
        mock_supabase_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_response
        
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            result = await client.delete_conversation('test_conv', 'test_user')
            
            assert result is True
    
    def test_supabase_client_disabled_when_no_credentials(self):
        """Test that Supabase client is disabled when no credentials are provided."""
        with patch.dict(os.environ, {}, clear=True):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            assert client.is_enabled() is False
    
    def test_supabase_client_enabled_when_credentials_provided(self, mock_supabase_client):
        """Test that Supabase client is enabled when credentials are provided."""
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test_key'
        }):
            from agent.memory.supabase_client import SupabaseClient
            
            client = SupabaseClient()
            assert client.is_enabled() is True

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 