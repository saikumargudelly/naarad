# Naarad AI Backend Dependencies

This document outlines the dependency management and compatibility for the Naarad AI Backend.

## Overview

The backend has been updated to use the latest compatible versions of all dependencies, with a focus on:

- **LangChain Ecosystem**: Latest v0.1.x series with full compatibility
- **FastAPI**: Latest v0.1.x series for modern web API development
- **Pydantic**: v2.x for advanced data validation
- **Python**: 3.8+ compatibility

## Dependency Categories

### Core Web Framework
- **FastAPI** (≥0.110.0): Modern, fast web framework for building APIs
- **Uvicorn** (≥0.27.0): Lightning-fast ASGI server
- **Python-multipart** (≥0.0.6): File upload support

### Rate Limiting
- **SlowAPI** (≥0.1.9): Rate limiting for API endpoints

### Data Validation & Settings
- **Pydantic** (≥2.11.0): Data validation using Python type annotations
- **Pydantic-settings** (≥2.9.0): Settings management using Pydantic
- **Pydantic-extra-types** (≥2.10.0): Additional Pydantic field types

### LangChain Ecosystem
- **LangChain** (≥0.1.20): Core LangChain framework
- **LangChain-core** (≥0.3.65): Core LangChain components
- **LangChain-community** (≥0.0.38): Community integrations
- **LangChain-groq** (≥0.3.2): Groq integration
- **LangChain-openai** (≥0.0.8): OpenAI integration
- **LangChain-text-splitters** (≥0.0.2): Text splitting utilities

### LLM Providers
- **Groq** (≥0.4.1): Groq API client
- **OpenAI** (≥1.88.0): OpenAI API client

### AI/ML Libraries
- **Sentence-transformers** (≥2.7.0): Sentence embeddings
- **Transformers** (≥4.52.0): Hugging Face transformers
- **Torch** (≥2.7.0): PyTorch deep learning framework
- **NumPy** (≥1.26.0): Numerical computing

### Database (Optional)
- **SQLAlchemy** (≥2.0.0): SQL toolkit and ORM
- **Alembic** (≥1.12.0): Database migration tool
- **Psycopg2-binary** (≥2.9.10): PostgreSQL adapter

### Authentication & Security
- **Python-jose** (≥3.5.0): JavaScript Object Signing and Encryption
- **Passlib** (≥1.7.4): Password hashing library

### Monitoring & Metrics
- **Prometheus-client** (≥0.22.0): Prometheus metrics

### Testing
- **Pytest** (≥8.4.0): Testing framework
- **Pytest-asyncio** (≥1.0.0): Async testing support
- **Pytest-cov** (≥6.2.0): Coverage reporting
- **Pytest-mock** (≥3.14.0): Mocking support

### Development Tools
- **Black** (≥23.0.0): Code formatter
- **Isort** (≥6.0.0): Import sorter
- **Flake8** (≥6.0.0): Linter
- **MyPy** (≥1.16.0): Type checker

## Installation

### Quick Start

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

Run the dependency test script to verify all installations:

```bash
python test_dependencies.py
```

This will:
- Test all required imports
- Verify version compatibility
- Provide installation guidance for missing dependencies

## Version Management

### Updating Dependencies

1. **Check current versions:**
   ```bash
   python test_dependencies.py
   ```

2. **Update to latest compatible versions:**
   ```bash
   ./update_deps.sh
   ```

3. **Verify updates:**
   ```bash
   python test_dependencies.py
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
   python test_dependencies.py
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
2. Run the test script (`test_dependencies.py`)
3. Review this documentation
4. Check individual package documentation

## Contributing

When adding new dependencies:

1. Add to appropriate requirements file
2. Update compatibility layer if needed
3. Add to test script
4. Update this documentation
5. Test with existing codebase 