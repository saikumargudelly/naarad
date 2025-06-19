#!/bin/bash

# Update pip
pip install --upgrade pip

# Install development tools
pip install --upgrade \
    black \
    isort \
    mypy \
    flake8 \
    pre-commit

# Install testing dependencies
pip install --upgrade \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    pytest-xdist \
    pytest-benchmark

# Install type stubs
pip install --upgrade \
    types-requests \
    types-python-dateutil \
    types-PyYAML \
    typing-extensions

# Install documentation tools
pip install --upgrade \
    mkdocs \
    mkdocs-material \
    "mkdocstrings[python]"

# Install code quality tools
pip install --upgrade \
    autoflake \
    bandit \
    mypy-extensions

# Install Jupyter for notebooks (optional)
pip install --upgrade \
    jupyter \
    ipython

# Install Pydantic development tools
pip install --upgrade \
    "pydantic-settings[cli]" \
    pydantic-extra-types

echo "Development dependencies have been updated successfully!"
