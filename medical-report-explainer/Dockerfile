# ============================================================
# Dockerfile — AI Medical Report Explainer (Backend)
# ============================================================
# Build context: medical-report-explainer/ directory
#
# From the REPO ROOT:
#   docker build -f medical-report-explainer/Dockerfile \
#     medical-report-explainer/ -t medexplain-api
#
# From INSIDE medical-report-explainer/:
#   docker build -t medexplain-api .
#
# Run locally (with a .env file):
#   docker run --env-file backend/.env -p 8000:8000 medexplain-api
# ============================================================

# ---- Stage 1: Builder — install deps ----
FROM python:3.11.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System packages needed to compile Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime — lean production image ----
FROM python:3.11.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tesseract binary location in Debian slim (used by pytesseract)
    TESSERACT_CMD=/usr/bin/tesseract \
    # Port default; Render overrides this with its own $PORT at runtime
    PORT=8000

WORKDIR /app

# Runtime OS dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR engine + English language data (required by pytesseract)
    tesseract-ocr \
    tesseract-ocr-eng \
    # Required by PyMuPDF for PDF page rendering to images
    libgl1 \
    libglib2.0-0 \
    # Required by Pillow for image processing
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source (backend + frontend)
COPY . .

# Ensure uploads directory exists (Render free tier: ephemeral storage)
RUN mkdir -p /app/backend/uploads/reports

# Train the ML model during build (skipped automatically if pkl already exists)
# Running via python script directly — works regardless of PYTHONPATH
RUN cd /app/backend && python -m ml.train

# Expose default port (Render ignores this and injects $PORT)
EXPOSE 8000

# Start command: cd into backend so relative imports and paths work correctly
# Shell form required so $PORT is expanded from Render's runtime environment
CMD ["sh", "-c", "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
