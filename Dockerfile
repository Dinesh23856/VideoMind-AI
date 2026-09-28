FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies first for better Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Application environment
ENV PYTHONPATH=/app
ENV SCRATCH_DIR=/tmp/media_proc
ENV PYTHONUNBUFFERED=1

# Create scratch directory
RUN mkdir -p /tmp/media_proc

# Render provides the PORT environment variable.
# 8000 is used only as a local fallback.
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]