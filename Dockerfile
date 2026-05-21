FROM python:3.10-slim

WORKDIR /app

# Install system deps for numba/scipy
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH so modules resolve correctly
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m appuser
USER appuser

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "daemon.py"]
