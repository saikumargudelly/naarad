#!/usr/bin/env python3
"""Test script to verify all dependencies are properly installed and compatible."""

import sys
import importlib
from typing import Dict, List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        # Try to get version, fallback to 'installed' if not available
        try:
            version = getattr(module, '__version__', 'installed')
            if version == 'installed':
                version = 'installed'
        except:
            version = 'installed'
        return True, version
    except ImportError as e:
        return False, str(e)

def test_dependencies() -> Dict[str, List[Tuple[str, bool, str]]]:
    """Test all required dependencies."""
    results = {
        'Core Web Framework': [
            ('fastapi', 'FastAPI web framework'),
            ('uvicorn', 'ASGI server'),
            ('python-multipart', 'File upload support'),

        ],
        'Rate Limiting': [
            ('slowapi', 'Rate limiting'),
        ],
        'Data Validation': [
            ('pydantic', 'Data validation'),
            ('pydantic_settings', 'Settings management'),
            ('pydantic_extra_types', 'Additional Pydantic types'),
        ],
        'LangChain Ecosystem': [
            ('langchain', 'LangChain core'),
            ('langchain_core', 'LangChain core components'),
            ('langchain_community', 'LangChain community integrations'),
            ('langchain_groq', 'Groq integration'),
            ('langchain_openai', 'OpenAI integration'),
        ],
        'LLM Providers': [
            ('groq', 'Groq client'),
            ('openai', 'OpenAI client'),
        ],
        'AI/ML Libraries': [
            ('numpy', 'Numerical computing'),
            ('torch', 'PyTorch'),
            ('transformers', 'Hugging Face transformers'),
        ],
        'Testing': [
            ('pytest', 'Testing framework'),
            ('pytest_asyncio', 'Async testing support'),
        ],
        'Development': [
            ('black', 'Code formatting'),
            ('isort', 'Import sorting'),
            ('flake8', 'Linting'),
        ]
    }
    
    test_results = {}
    
    for category, modules in results.items():
        category_results = []
        for module_name, description in modules:
            success, version_or_error = test_import(module_name)
            category_results.append((description, success, version_or_error))
        test_results[category] = category_results
    
    return test_results

def print_results(results: Dict[str, List[Tuple[str, bool, str]]]):
    """Print test results in a formatted way."""
    print("🔍 Dependency Compatibility Test Results")
    print("=" * 50)
    
    all_passed = True
    
    for category, modules in results.items():
        print(f"\n📦 {category}")
        print("-" * len(category))
        
        for description, success, version_or_error in modules:
            if success:
                print(f"  ✅ {description}: {version_or_error}")
            else:
                print(f"  ❌ {description}: {version_or_error}")
                all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All dependencies are properly installed!")
        return True
    else:
        print("⚠️  Some dependencies failed to import.")
        print("Run: ./update_deps.sh to install missing dependencies")
        return False

def test_compatibility():
    """Test compatibility between different versions."""
    print("\n🔧 Compatibility Tests")
    print("-" * 30)
    
    # Test Pydantic version
    try:
        import pydantic
        version = pydantic.__version__
        if version.startswith('2.'):
            print(f"✅ Pydantic v2.x detected: {version}")
        elif version.startswith('1.'):
            print(f"⚠️  Pydantic v1.x detected: {version} (v2.x recommended)")
        else:
            print(f"❓ Unknown Pydantic version: {version}")
    except ImportError:
        print("❌ Pydantic not installed")
    
    # Test LangChain version
    try:
        import langchain
        version = langchain.__version__
        if version.startswith('0.3.'):
            print(f"✅ LangChain v0.3.x detected: {version}")
        elif version.startswith('0.1.'):
            print(f"✅ LangChain v0.1.x detected: {version}")
        else:
            print(f"❓ Unknown LangChain version: {version}")
    except ImportError:
        print("❌ LangChain not installed")
    
    # Test FastAPI version
    try:
        import fastapi
        version = fastapi.__version__
        if version.startswith('0.1'):
            print(f"✅ FastAPI v0.1.x detected: {version}")
        else:
            print(f"❓ Unknown FastAPI version: {version}")
    except ImportError:
        print("❌ FastAPI not installed")

def main():
    """Main test function."""
    print("🚀 Testing Naarad AI Backend Dependencies")
    print("=" * 50)
    
    # Test basic imports
    results = test_dependencies()
    success = print_results(results)
    
    # Test compatibility
    test_compatibility()
    
    if success:
        print("\n🎯 All tests passed! Your backend is ready to run.")
        print("\nTo start the server:")
        print("  cd backend")
        print("  source venv/bin/activate")
        print("  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n🔧 Please install missing dependencies and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 