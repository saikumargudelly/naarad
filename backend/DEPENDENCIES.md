# Naarad AI Assistant - Dependencies Guide

## 🎯 Overview

This document provides comprehensive information about the dependencies used in the Naarad AI Assistant project, including installation, compatibility, and troubleshooting.

## 📦 Core Dependencies

### Web Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Python-multipart**: File upload support

### Rate Limiting
- **SlowAPI**: Rate limiting middleware for FastAPI

### Async and Networking
- **AnyIO**: Async I/O library
- **AIOHTTP**: Async HTTP client/server
- **HTTPX**: Modern HTTP client

### Data Validation and Settings
- **Pydantic**: Data validation using Python type annotations
- **Pydantic-settings**: Settings management using Pydantic
- **Pydantic-extra-types**: Additional Pydantic field types

### Environment and Configuration
- **Python-dotenv**: Environment variable management

### LangChain Ecosystem
- **LangChain**: Framework for developing applications with LLMs
- **LangChain-core**: Core LangChain functionality
- **LangChain-community**: Community integrations
- **LangChain-groq**: Groq integration
- **LangChain-openai**: OpenAI integration
- **LangChain-text-splitters**: Text splitting utilities
- **LangChain-hub**: Model and prompt hub
- **LangChain-anthropic**: Anthropic integration
- **LangChain-experimental**: Experimental features
- **LangChain-google-genai**: Google Generative AI integration
- **LangChain-fireworks**: Fireworks AI integration
- **LangChain-mistralai**: Mistral AI integration
- **LangChain-nomic**: Nomic integration
- **LangChain-together**: Together AI integration
- **LangChain-voyageai**: Voyage AI integration
- **LangChain-pinecone**: Pinecone vector store integration
- **LangChain-chroma**: Chroma vector store integration
- **LangChain-elasticsearch**: Elasticsearch integration
- **LangChain-qdrant**: Qdrant vector store integration
- **LangChain-weaviate**: Weaviate vector store integration
- **LangChain-astradb**: Astra DB integration
- **LangChain-mongodb**: MongoDB integration
- **LangChain-aws**: AWS integrations
- **LangChain-upstage**: Upstage integration
- **LangChain-zhipuai**: Zhipu AI integration
- **LangChain-baidu-qianfan**: Baidu Qianfan integration
- **LangChain-box**: Box integration
- **LangChain-clarifai**: Clarifai integration
- **LangChain-cohere**: Cohere integration
- **LangChain-couchbase**: Couchbase integration
- **LangChain-google-vertexai**: Google Vertex AI integration
- **LangChain-ibm**: IBM integration
- **LangChain-milvus**: Milvus integration
- **LangChain-postgres**: PostgreSQL integration
- **LangChain-robocorp**: Robocorp integration
- **LangChain-exa**: Exa integration
- **LangChain-google-community**: Google Community integrations
- **LangChain-huggingface**: Hugging Face integration
- **LangChain-ollama**: Ollama integration
- **LangChain-nvidia-ai-endpoints**: NVIDIA AI Endpoints integration
- **LangChain-yandex**: Yandex integration
- **LangChain-redis**: Redis integration
- **LangChain-community[graph_vector_stores]**: Graph vector stores

### LLM Providers
- **Groq**: Fast LLM inference
- **OpenAI**: OpenAI API integration

### AI/ML Libraries
- **Sentence-transformers**: Sentence embeddings
- **Transformers**: Hugging Face transformers
- **Torch**: PyTorch deep learning framework
- **NumPy**: Numerical computing

### Database
- **SQLAlchemy**: SQL toolkit and ORM
- **Alembic**: Database migration tool
- **Psycopg2-binary**: PostgreSQL adapter
- **Greenlet**: Lightweight coroutines

### Authentication and Security
- **Python-jose[cryptography]**: JWT implementation
- **Passlib[bcrypt]**: Password hashing

### Monitoring and Metrics
- **Prometheus-client**: Prometheus metrics

### HTTP Requests
- **Requests**: HTTP library

### Testing
- **Pytest**: Testing framework
- **Pytest-asyncio**: Async testing support
- **Pytest-cov**: Coverage reporting
- **Pytest-mock**: Mocking support

### Code Quality and Development
- **Black**: Code formatter
- **Isort**: Import sorter
- **Flake8**: Linter
- **MyPy**: Type checker

### Documentation
- **MkDocs**: Documentation generator
- **MkDocs-material**: Material theme for MkDocs
- **Mkdocstrings**: Automatic API documentation
- **Mkdocstrings-python**: Python support for mkdocstrings

### Utilities
- **Python-dateutil**: Date utilities
- **Tenacity**: Retry library
- **Tiktoken**: Tokenizer for OpenAI models

## 🚀 Installation

### Automated Installation

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Run the update script:**
   ```bash
   ./update_deps.sh
   ```

3. **For development dependencies:**
   ```bash
   ./update_deps.sh --dev
   ```

### Manual Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies (optional):**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Compatibility Layer

The backend includes a comprehensive compatibility layer (`agent/compat.py`) that handles:

### LangChain Version Compatibility
- **v1.x**: Legacy LangChain support
- **v2.x/v3.x**: Current LangChain support
- Automatic import resolution based on installed version

### Pydantic Version Compatibility
- **v1.x**: Legacy Pydantic support
- **v2.x**: Current Pydantic support with ConfigDict
- Automatic field validator resolution

### FastAPI Compatibility
- Version-agnostic FastAPI imports
- Middleware compatibility
- Rate limiting integration

## Testing Dependencies

To verify all installations, run the test suite:

```bash
pytest tests/ -v
```

This will:
- Test all required imports
- Verify version compatibility
- Provide installation guidance for missing dependencies

## Version Management

### Updating Dependencies

1. **Check current versions:**
   ```bash
   pip list
   ```

2. **Update to latest compatible versions:**
   ```bash
   ./update_deps.sh
   ```

3. **Verify updates:**
   ```bash
   pytest tests/ -v
   ```

### Pinning Specific Versions

To pin specific versions, edit `requirements.txt`:

```txt
# Example: Pin specific version
fastapi==0.110.0
langchain==0.1.20
```

### Development vs Production

- **Production**: Use `requirements.txt` for minimal dependencies
- **Development**: Use `requirements-dev.txt` for additional tools

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

2. **Version Conflicts**
   ```bash
   # Clean install
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

3. **LangChain Compatibility**
   ```bash
   # Check LangChain version
   python -c "import langchain; print(langchain.__version__)"
   ```

### Environment Issues

1. **Virtual Environment**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Python Version**
   ```bash
   # Check Python version (requires 3.8+)
   python --version
   ```

## Security Considerations

- All dependencies are pinned to minimum versions for security
- Regular updates recommended for security patches
- Development dependencies separated from production
- Rate limiting enabled by default

## Performance Optimization

- **Uvicorn**: Configured with standard extras for performance
- **Async Support**: Full async/await support throughout
- **Caching**: Optional caching layer support
- **Monitoring**: Prometheus metrics integration

## Migration Guide

### From Previous Versions

1. **Backup current environment:**
   ```bash
   pip freeze > requirements_backup.txt
   ```

2. **Update dependencies:**
   ```bash
   ./update_deps.sh
   ```

3. **Test compatibility:**
   ```bash
   pytest tests/ -v
   ```

4. **Run tests:**
   ```bash
   pytest tests/
   ```

### Breaking Changes

- **LangChain**: Updated to v0.1.x series
- **Pydantic**: Updated to v2.x series
- **FastAPI**: Updated to v0.1.x series

## Support

For dependency-related issues:

1. Check the compatibility layer (`agent/compat.py`)
2. Run the test suite (`pytest tests/`)
3. Review this documentation
4. Check individual package documentation

## Contributing

When adding new dependencies:

1. Add to appropriate requirements file
2. Update compatibility layer if needed
3. Add to test suite
4. Update this documentation
5. Test with existing codebase 