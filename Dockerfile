FROM python:3.12-slim

# Set environment variables for Python and Hugging Face model cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

# Application directory
WORKDIR /app

# Create non-root application user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/.cache/huggingface /app/data && \
    chown -R appuser:appuser /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (optimizes Docker layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies
RUN uv sync --frozen --no-dev

# Pre-download and cache ML models for offline resilience and fast cold-starts
COPY scripts/download_models.py ./scripts/download_models.py
RUN uv run python scripts/download_models.py

# Copy application code
COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY brand_vocab ./brand_vocab

# Ensure app ownership belongs to appuser
RUN chown -R appuser:appuser /app

# Switch to unprivileged user
USER appuser

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501"]