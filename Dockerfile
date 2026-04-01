# =========================================================
# Hugging Face Spaces — Pipeline Debugger (OpenEnv)
# =========================================================
# Build:  docker build -t pipeline-debugger .
# Run:    docker run -p 7860:7860 pipeline-debugger
# =========================================================

FROM python:3.11-slim

# HF Spaces metadata
LABEL org.opencontainers.image.title="Pipeline Debugger OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv environment for AI pipeline debugging"
LABEL tags="openenv"

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]