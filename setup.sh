#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Setting up Naarad AI Assistant...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}❌ Python 3 is required but not installed. Please install Python 3.8+ and try again.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}❌ Node.js is required for the frontend but not installed. Please install Node.js 16+ and try again.${NC}"
    exit 1
fi

# Create and activate virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${NC}
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}
cd backend
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
echo -e "${GREEN}Setting up environment variables...${NC}
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}Please update the .env file with your API keys.${NC}
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

# Install Node.js dependencies
echo -e "${GREEN}Setting up frontend dependencies...${NC}
cd ../frontend
npm install

# Create .env file for frontend
if [ ! -f .env ]; then
    echo "REACT_APP_API_URL=http://localhost:8000" > .env
    echo -e "${GREEN}Created frontend .env file.${NC}
else
    echo -e "${GREEN}Frontend .env file already exists.${NC}"
fi

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\nTo start the backend server, run:"
echo -e "  cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo -e "\nTo start the frontend development server, run:"
echo -e "  cd frontend && npm start"
echo -e "\nThen open http://localhost:3000 in your browser to access Naarad AI Assistant.${NC}"
