#!/bin/bash

# Update pip
pip install --upgrade pip

# Install core dependencies
pip install --upgrade \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    python-dotenv \
    pydantic \
    pydantic-settings \
    pydantic-core \
    pydantic-extra-types \
    email-validator

# Install AI/ML dependencies
pip install --upgrade \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    tiktoken \
    tqdm

# Install LangChain stack
pip install --upgrade \
    langchain \
    langchain-community \
    langchain-core \
    langchain-openai \
    langchain-text-splitters

# Install database dependencies
pip install --upgrade \
    SQLAlchemy \
    alembic \
    psycopg2-binary

# Install HTTP/API dependencies
pip install --upgrade \
    httpx \
    requests \
    aiohttp \
    python-jose[cryptography] \
    yarl \
    websockets

# Install utility dependencies
pip install --upgrade \
    anyio \
    slowapi \
    PyYAML \
    regex \
    rsa \
    shellingham \
    sniffio \
    starlette \
    tenacity \
    typer \
    typing-extensions \
    ujson \
    urllib3 \
    uvloop \
    watchfiles \
    orjson

# Install monitoring dependencies
pip install --upgrade prometheus-client

echo "All dependencies have been updated successfully!"
