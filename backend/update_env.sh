#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting project update..."

# Navigate to the backend directory
cd "$(dirname "$0")"

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔍 Found existing virtual environment. Creating a backup..."
    mv venv venv_backup_$(date +%Y%m%d_%H%M%S)
fi

# Create a new virtual environment
echo "🐍 Creating new virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✨ Environment update complete!"
echo "   To activate the new environment, run: source venv/bin/activate"
echo "   To start the application: uvicorn main:app --reload"
