#!/usr/bin/env python3
"""
Environment setup script for Naarad AI Assistant.

This script helps set up the required environment variables for testing.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables for testing."""
    
    # Get the backend directory
    backend_dir = Path(__file__).parent
    env_file = backend_dir / ".env"
    
    print("🔧 Setting up environment for Naarad AI Assistant...")
    
    # Check if .env file exists
    if env_file.exists():
        print(f"✅ .env file found at {env_file}")
    else:
        print(f"⚠️  .env file not found at {env_file}")
        print("   Creating a basic .env file with required variables...")
        
        # Create basic .env content
        env_content = """# Naarad AI Assistant Environment Configuration

# LLM Provider Configuration
LLM_PROVIDER=groq
CHAT_MODEL=llama3-8b-8192
REASONING_MODEL=llama3-8b-8192
VISION_MODEL=llava-1.5-7b
EMBEDDING_MODEL=text-embedding-3-small

# API Keys (Add your actual keys here)
GROQ_API_KEY=your_groq_api_key_here
BRAVE_API_KEY=your_brave_api_key_here

# Security
SECRET_KEY=your_secret_key_here_minimum_32_characters

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
ENVIRONMENT=development

# CORS Configuration
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Database (Optional)
DATABASE_URL=sqlite:///./naarad.db

# Rate Limiting
RATE_LIMIT=100/minute

# Token Configuration
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# File Upload
UPLOAD_FOLDER=uploads
MAX_UPLOAD_SIZE=10485760
ALLOWED_EXTENSIONS=["jpg","jpeg","png","gif","pdf"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/naarad.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9100

# Caching
ENABLE_CACHE=true
CACHE_TTL=300
"""
        
        # Write the .env file
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Created .env file at {env_file}")
    
    # Check for required environment variables
    required_vars = [
        'GROQ_API_KEY',
        'BRAVE_API_KEY', 
        'SECRET_KEY',
        'LLM_PROVIDER',
        'CHAT_MODEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please add these to your .env file")
        print("   You can get API keys from:")
        print("   - Groq: https://console.groq.com/")
        print("   - Brave Search: https://api.search.brave.com/")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def main():
    """Main function."""
    success = setup_environment()
    
    if success:
        print("\n🎉 Environment setup completed successfully!")
        print("   You can now run the tests with: python run_all_tests.py")
    else:
        print("\n⚠️  Environment setup incomplete.")
        print("   Please configure the missing environment variables and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 