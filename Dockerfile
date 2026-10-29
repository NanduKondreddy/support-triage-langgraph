# Multi-Domain Support Triage Agent - LangGraph Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code and data corpus
COPY code/ /app/code/
COPY data/ /app/data/
COPY support_issues/ /app/support_issues/

ENV PYTHONPATH="/app/code"
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI service with Uvicorn using --app-dir
CMD ["uvicorn", "api:app", "--app-dir", "code", "--host", "0.0.0.0", "--port", "8000"]
