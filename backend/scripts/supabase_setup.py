#!/usr/bin/env python3
"""
Supabase Setup Script for Naarad AI Assistant

This script helps you set up and configure Supabase integration for the Naarad AI Assistant.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    from supabase import create_client, Client
    from agent.memory.supabase_client import SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("❌ Supabase library not installed. Run: pip install supabase")

def check_environment() -> Dict[str, Any]:
    """Check the current environment configuration."""
    print("🔍 Checking environment configuration...")
    
    config = {
        'supabase_url': os.getenv('SUPABASE_URL'),
        'supabase_key': os.getenv('SUPABASE_KEY'),
        'database_url': os.getenv('DATABASE_URL'),
        'supabase_available': SUPABASE_AVAILABLE
    }
    
    print(f"✅ Supabase URL: {'Set' if config['supabase_url'] else 'Not set'}")
    print(f"✅ Supabase Key: {'Set' if config['supabase_key'] else 'Not set'}")
    print(f"✅ Database URL: {config['database_url'] or 'Not set'}")
    print(f"✅ Supabase Library: {'Available' if config['supabase_available'] else 'Not available'}")
    
    return config

def test_supabase_connection(supabase_url: str, supabase_key: str) -> bool:
    """Test the Supabase connection."""
    print("🔗 Testing Supabase connection...")
    
    try:
        client = create_client(supabase_url, supabase_key)
        
        # Test basic connection by trying to access a table
        # This will fail if the table doesn't exist, but that's expected
        try:
            response = client.table("conversation_memories").select("count").limit(1).execute()
            print("✅ Supabase connection successful!")
            return True
        except Exception as e:
            if "relation" in str(e).lower() and "does not exist" in str(e).lower():
                print("⚠️  Supabase connection successful, but 'conversation_memories' table doesn't exist yet.")
                print("   Run the SQL setup script in your Supabase dashboard.")
                return True
            else:
                print(f"❌ Supabase connection failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        return False

def create_env_template():
    """Create a .env template with Supabase configuration."""
    print("📝 Creating .env template...")
    
    env_template = """# =============================================================================
# Naarad AI Assistant - Environment Configuration
# =============================================================================

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
APP_NAME=Naarad AI Assistant
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
HOST=0.0.0.0
PORT=8000
RELOAD=true
WORKERS=1

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_PREFIX=/api
API_V1_STR=/api/v1

# =============================================================================
# LLM PROVIDER API KEYS
# =============================================================================
# Groq API (Primary LLM Provider)
GROQ_API_KEY=your_groq_api_key_here

# OpenRouter API (Alternative LLM Provider)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Together.ai API (Alternative LLM Provider)
TOGETHER_API_KEY=your_together_api_key_here

# =============================================================================
# EXTERNAL SERVICE API KEYS
# =============================================================================
# Brave Search API for web search functionality
BRAVE_API_KEY=your_brave_search_api_key_here

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================
# Default LLM Provider
LLM_PROVIDER=groq

# Model Names
REASONING_MODEL=mixtral-8x7b-instruct-v0.1
CHAT_MODEL=mixtral-8x7b-instruct-v0.1
VISION_MODEL=llava-1.5-7b
EMBEDDING_MODEL=text-embedding-3-small

# Model Parameters
MODEL_TEMPERATURE=0.2
MODEL_MAX_TOKENS=8192
MODEL_TOP_P=0.95
MODEL_FREQUENCY_PENALTY=0.0
MODEL_PRESENCE_PENALTY=0.0

# Request Settings
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=1.0

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# SQLite (Default for development)
DATABASE_URL=sqlite:///./naarad_memory.db

# PostgreSQL (For production)
# DATABASE_URL=postgresql://username:password@localhost:5432/naarad_db

# =============================================================================
# SECURITY SETTINGS
# =============================================================================
SECRET_KEY=your-super-secret-key-here-make-it-at-least-32-characters-long
ACCESS_TOKEN_EXPIRE_MINUTES=10080
REFRESH_TOKEN_EXPIRE_DAYS=30
ALGORITHM=HS256

# CORS Origins
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000","http://127.0.0.1:3000","http://127.0.0.1:8000"]

# Rate Limiting
RATE_LIMIT=100/minute
RATE_LIMIT_WINDOW=60

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/naarad.log

# =============================================================================
# FILE STORAGE
# =============================================================================
UPLOAD_FOLDER=uploads
MAX_UPLOAD_SIZE=10485760
ALLOWED_EXTENSIONS=["jpg","jpeg","png","gif","pdf"]

# =============================================================================
# CACHING
# =============================================================================
ENABLE_CACHE=true
CACHE_TTL=300

# =============================================================================
# MONITORING
# =============================================================================
ENABLE_METRICS=true
METRICS_PORT=9100

# =============================================================================
# SUPABASE CONFIGURATION
# =============================================================================
# Get these values from your Supabase project dashboard
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================
# Set to true for development, false for production
DEBUG=true
ENVIRONMENT=development
"""
    
    env_file = backend_dir / ".env"
    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Skipping .env file creation.")
            return
    
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print(f"✅ Created .env template at {env_file}")
    print("📝 Please edit the file and add your actual API keys and Supabase credentials.")

def print_setup_instructions():
    """Print setup instructions for Supabase."""
    print("\n" + "="*80)
    print("🚀 SUPABASE SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. 📋 Create a Supabase Project:")
    print("   - Go to https://supabase.com")
    print("   - Sign up or log in")
    print("   - Create a new project")
    print("   - Wait for the project to be ready")
    
    print("\n2. 🔑 Get Your Credentials:")
    print("   - Go to Settings > API in your Supabase dashboard")
    print("   - Copy the 'Project URL' (SUPABASE_URL)")
    print("   - Copy the 'anon public' key (SUPABASE_KEY)")
    
    print("\n3. 📝 Configure Environment:")
    print("   - Edit the .env file in the backend directory")
    print("   - Set SUPABASE_URL and SUPABASE_KEY with your values")
    
    print("\n4. 🗄️  Set Up Database Tables:")
    print("   - Go to SQL Editor in your Supabase dashboard")
    print("   - Copy and paste the contents of scripts/supabase_setup.sql")
    print("   - Run the SQL script to create the necessary tables")
    
    print("\n5. 🧪 Test the Integration:")
    print("   - Run this script again to test the connection")
    print("   - Start your application and test conversation storage")
    
    print("\n6. 🔒 Security Notes:")
    print("   - The anon key is safe to use in client-side code")
    print("   - Row Level Security (RLS) is enabled by default")
    print("   - Users can only access their own conversations")
    
    print("\n" + "="*80)

async def test_memory_manager():
    """Test the memory manager with Supabase integration."""
    print("🧪 Testing memory manager integration...")
    
    try:
        from agent.memory.memory_manager import memory_manager
        
        # Test conversation operations
        test_conversation_id = "test_supabase_integration"
        test_user_id = "test_user"
        test_messages = [
            {"role": "user", "content": "Hello, this is a test message"},
            {"role": "assistant", "content": "Hi! This is a test response"}
        ]
        
        # Test saving conversation
        print("   Testing conversation save...")
        result = await memory_manager.save_conversation(
            test_conversation_id, test_messages, test_user_id
        )
        
        if result:
            print("   ✅ Conversation saved successfully")
            
            # Test retrieving conversation
            print("   Testing conversation retrieval...")
            retrieved = await memory_manager.get_conversation(test_conversation_id, test_user_id)
            
            if retrieved and retrieved.get('messages') == test_messages:
                print("   ✅ Conversation retrieved successfully")
                
                # Test listing conversations
                print("   Testing conversation listing...")
                conversations = await memory_manager.list_conversations(test_user_id)
                
                if conversations:
                    print(f"   ✅ Found {len(conversations)} conversations")
                    
                    # Test deleting conversation
                    print("   Testing conversation deletion...")
                    deleted = await memory_manager.delete_conversation(test_conversation_id, test_user_id)
                    
                    if deleted:
                        print("   ✅ Conversation deleted successfully")
                    else:
                        print("   ❌ Failed to delete conversation")
                else:
                    print("   ⚠️  No conversations found in listing")
            else:
                print("   ❌ Failed to retrieve conversation")
        else:
            print("   ❌ Failed to save conversation")
            
    except Exception as e:
        print(f"   ❌ Memory manager test failed: {e}")

def main():
    """Main function to run the setup script."""
    print("🚀 Naarad AI Assistant - Supabase Setup Script")
    print("="*50)
    
    # Check environment
    config = check_environment()
    
    if not config['supabase_available']:
        print("\n❌ Supabase library not available. Please install it first:")
        print("   pip install supabase")
        return
    
    if not config['supabase_url'] or not config['supabase_key']:
        print("\n⚠️  Supabase credentials not configured.")
        print("   Creating .env template...")
        create_env_template()
        print_setup_instructions()
        return
    
    # Test connection
    if test_supabase_connection(config['supabase_url'], config['supabase_key']):
        print("\n✅ Supabase is properly configured!")
        
        # Test memory manager
        print("\n🧪 Testing memory manager integration...")
        asyncio.run(test_memory_manager())
        
        print("\n🎉 Supabase integration is ready to use!")
        print("   You can now start your application and conversations will be stored in Supabase.")
    else:
        print("\n❌ Supabase connection failed.")
        print("   Please check your credentials and try again.")

if __name__ == "__main__":
    main() 