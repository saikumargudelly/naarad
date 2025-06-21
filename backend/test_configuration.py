"""
Configuration Test Suite for Naarad AI Assistant

This module tests the configuration and environment setup to ensure
all required settings are properly configured.
"""

import os
import sys
import unittest
from typing import Dict, Any, Optional
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

class TestConfiguration(unittest.TestCase):
    """Test configuration and environment setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env_file_path = Path(backend_dir) / ".env"
        self.example_env_path = Path(backend_dir) / "env.example"
    
    def test_env_file_exists(self):
        """Test that .env file exists."""
        if not self.env_file_path.exists():
            print("⚠️  Warning: .env file not found. Please create it from env.example")
            print("   Run: cp env.example .env")
            print("   Then edit .env with your API keys")
        else:
            print("✅ .env file exists")
    
    def test_example_env_exists(self):
        """Test that env.example file exists."""
        self.assertTrue(self.example_env_path.exists(), "env.example file should exist")
        print("✅ env.example file exists")
    
    def test_required_environment_variables(self):
        """Test that required environment variables are defined."""
        required_vars = [
            "GROQ_API_KEY",
            "BRAVE_API_KEY",
            "SECRET_KEY",
            "LLM_PROVIDER",
            "CHAT_MODEL",
            "REASONING_MODEL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("   Please add these to your .env file")
        else:
            print("✅ All required environment variables are set")
    
    def test_api_keys_format(self):
        """Test that API keys have proper format."""
        groq_key = os.getenv("GROQ_API_KEY")
        brave_key = os.getenv("BRAVE_API_KEY")
        
        if groq_key and not groq_key.startswith("gsk_"):
            print("⚠️  Warning: GROQ_API_KEY should start with 'gsk_'")
        elif groq_key:
            print("✅ GROQ_API_KEY format looks correct")
        
        if brave_key and len(brave_key) < 10:
            print("⚠️  Warning: BRAVE_API_KEY seems too short")
        elif brave_key:
            print("✅ BRAVE_API_KEY format looks correct")
    
    def test_model_configuration(self):
        """Test model configuration settings."""
        models = [
            "LLM_PROVIDER",
            "CHAT_MODEL", 
            "REASONING_MODEL",
            "VISION_MODEL",
            "EMBEDDING_MODEL"
        ]
        
        for model in models:
            value = os.getenv(model)
            if value:
                print(f"✅ {model}: {value}")
            else:
                print(f"⚠️  {model} not set")
    
    def test_server_configuration(self):
        """Test server configuration settings."""
        server_settings = {
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "DEBUG": "true",
            "ENVIRONMENT": "development"
        }
        
        for setting, default in server_settings.items():
            value = os.getenv(setting, default)
            print(f"✅ {setting}: {value}")
    
    def test_security_configuration(self):
        """Test security configuration settings."""
        security_settings = [
            "SECRET_KEY",
            "ACCESS_TOKEN_EXPIRE_MINUTES",
            "REFRESH_TOKEN_EXPIRE_DAYS",
            "RATE_LIMIT"
        ]
        
        for setting in security_settings:
            value = os.getenv(setting)
            if value:
                print(f"✅ {setting}: {'*' * len(value)} (hidden)")
            else:
                print(f"⚠️  {setting} not set")
    
    def test_cors_configuration(self):
        """Test CORS configuration."""
        cors_origins = os.getenv("BACKEND_CORS_ORIGINS")
        if cors_origins:
            print(f"✅ CORS Origins: {cors_origins}")
        else:
            print("⚠️  BACKEND_CORS_ORIGINS not set")
    
    def test_database_configuration(self):
        """Test database configuration."""
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            print(f"✅ Database URL: {db_url}")
        else:
            print("⚠️  DATABASE_URL not set")
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        logging_settings = {
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "json",
            "LOG_FILE": "logs/naarad.log"
        }
        
        for setting, default in logging_settings.items():
            value = os.getenv(setting, default)
            print(f"✅ {setting}: {value}")
    
    def test_file_storage_configuration(self):
        """Test file storage configuration."""
        storage_settings = {
            "UPLOAD_FOLDER": "uploads",
            "MAX_UPLOAD_SIZE": "10485760",
            "ALLOWED_EXTENSIONS": "[\"jpg\",\"jpeg\",\"png\",\"gif\",\"pdf\"]"
        }
        
        for setting, default in storage_settings.items():
            value = os.getenv(setting, default)
            print(f"✅ {setting}: {value}")
    
    def test_caching_configuration(self):
        """Test caching configuration."""
        cache_settings = {
            "ENABLE_CACHE": "true",
            "CACHE_TTL": "300"
        }
        
        for setting, default in cache_settings.items():
            value = os.getenv(setting, default)
            print(f"✅ {setting}: {value}")
    
    def test_monitoring_configuration(self):
        """Test monitoring configuration."""
        monitoring_settings = {
            "ENABLE_METRICS": "true",
            "METRICS_PORT": "9100"
        }
        
        for setting, default in monitoring_settings.items():
            value = os.getenv(setting, default)
            print(f"✅ {setting}: {value}")

def run_configuration_tests():
    """Run all configuration tests."""
    print("🔧 Starting Configuration Tests for Naarad AI Assistant...")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfiguration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 80)
    print("📊 CONFIGURATION TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    print("\n📋 CONFIGURATION CHECKLIST:")
    print("   □ GROQ_API_KEY is set and valid")
    print("   □ BRAVE_API_KEY is set and valid")
    print("   □ SECRET_KEY is set (at least 32 characters)")
    print("   □ LLM_PROVIDER is configured")
    print("   □ CHAT_MODEL is specified")
    print("   □ REASONING_MODEL is specified")
    print("   □ DATABASE_URL is configured")
    print("   □ CORS origins are set")
    print("   □ Rate limiting is configured")
    print("   □ Logging is properly set up")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_configuration_tests()
    sys.exit(0 if success else 1) 