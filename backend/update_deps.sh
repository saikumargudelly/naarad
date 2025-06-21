#!/bin/bash

# Update dependencies script for Naarad AI Backend
# This script updates all dependencies to their latest compatible versions

set -e

echo "🚀 Updating Naarad AI Backend dependencies..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install/upgrade core dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements.txt --upgrade

# Install development dependencies (optional)
if [ "$1" = "--dev" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -r requirements-dev.txt --upgrade
fi

# Verify installations
echo "✅ Verifying installations..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import fastapi
    print(f'FastAPI version: {fastapi.__version__}')
except ImportError:
    print('FastAPI: Not installed')

try:
    import langchain
    print(f'LangChain version: {langchain.__version__}')
except ImportError:
    print('LangChain: Not installed')

try:
    import pydantic
    print(f'Pydantic version: {pydantic.__version__}')
except ImportError:
    print('Pydantic: Not installed')

try:
    import slowapi
    print(f'SlowAPI version: {slowapi.__version__}')
except ImportError:
    print('SlowAPI: Not installed')

try:
    import uvicorn
    print(f'Uvicorn version: {uvicorn.__version__}')
except ImportError:
    print('Uvicorn: Not installed')
"

echo "🎉 Dependencies updated successfully!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To run tests:"
echo "  pytest tests/"
