# Naarad - Your AI Companion 😺

Naarad is a curious, AI assistant that helps you with tasks, answers questions, and understands images. Built with FastAPI, LangChain, and modern AI models.

## 🚀 Features

- **Chat Interface**: Natural conversations with a friendly AI companion
- **Multimodal Inputs**: Support for both text and images
- **Web Search**: Real-time information retrieval using Brave Search API
- **Image Understanding**: Analyze and describe images using LLaVA
- **Modular Architecture**: Easy to extend with new tools and capabilities

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: LangChain, OpenRouter, Together.ai
- **Frontend**: React.js with Tailwind CSS
- **Search**: Brave Search API
- **Storage**: Supabase (optional)

## 🏗️ Project Structure

```
naarad/
├── backend/               # FastAPI backend
│   ├── agent/             # LangChain agent and tools
│   ├── llm/               # LLM configurations
│   ├── routers/           # API routes
│   ├── tools/             # Custom tools (search, vision, etc.)
│   ├── main.py            # FastAPI application
│   └── requirements.txt   # Python dependencies
└── frontend/              # React frontend (coming soon)
    ├── public/
    └── src/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- API keys for OpenRouter, Together.ai, and Brave Search

### Backend Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd naarad
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your API keys.

5. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

6. The API will be available at `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`
   - Redoc: `http://localhost:8000/redoc`

### Frontend Setup (Coming Soon)

```bash
cd frontend
npm install
npm start
```

## 🌐 API Endpoints

- `GET /` - Health check and API info
- `POST /api/chat` - Chat with Naarad
- `GET /api/health` - Service health check

## 🤖 Using the Chat API

Send a POST request to `/api/chat` with the following JSON body:

```json
{
  "message": "Tell me about cats",
  "images": [],
  "conversation_id": "optional-conversation-id",
  "chat_history": []
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using FastAPI, LangChain, and OpenRouter
- Thanks to all the open-source projects that made this possible
