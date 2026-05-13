# ─────────────────────────────────────────────────────────────────
# Stage 1: Builder
#   Install dependencies into a clean layer so the final image
#   does not contain pip cache or build tools.
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Keeps Python from writing .pyc files and enables stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Copy requirements and install into a target directory
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ─────────────────────────────────────────────────────────────────
# Stage 2: Runtime
#   Lean image — only production files, no build tools.
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser/app

# Copy installed packages from builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source code and trained model
COPY app/        ./app/
COPY model/      ./model/

# Give ownership to non-root user
RUN chown -R appuser:appuser /home/appuser/app

USER appuser

# Document the port (Render / Docker reads this)
EXPOSE 8000

# Health check — Render will poll this to verify the container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start FastAPI with uvicorn
#   --host 0.0.0.0  → listen on all interfaces (required inside a container)
#   --port $PORT    → use the PORT env var (Render sets this automatically)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
