# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system deps for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend FastAPI code and model artifacts
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/

# Set environment variable for model path
ENV MODEL_PATH=/app/artifacts/model_data.joblib

# Expose FastAPI port
EXPOSE 8000

# Simple healthcheck using curl (works on Windows and Linux)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Run FastAPI (module: backend.server)
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]