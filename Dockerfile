FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CAR_SCAN_ROOT=/app \
    CAR_SCAN_OUTPUT_DIR=/app/scan/data \
    CAR_SCAN_WEB_HOST=0.0.0.0 \
    CAR_SCAN_WEB_PORT=8000 \
    CAR_SCAN_COMPUTE=auto \
    PYTHONPATH=/app:/app/src \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TESSDATA_PREFIX=/app/tools/tesseract/tessdata

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tesseract-ocr \
        tesseract-ocr-tha \
        tesseract-ocr-lao \
        nodejs \
        openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements-docker.txt \
    && (pip uninstall -y opencv-python || true) \
    && pip install "opencv-python-headless>=4.10,<5"

COPY prisma /app/prisma
COPY . /app

RUN python -m prisma generate --schema /app/prisma/schema.prisma \
    && mkdir -p /app/tools/tesseract/tessdata /app/scan/data \
    && for pack in tha.traineddata lao.traineddata; do \
         if [ ! -f "/app/tools/tesseract/tessdata/${pack}" ]; then \
           found="$(find /usr/share -name "${pack}" 2>/dev/null | head -n 1)"; \
           if [ -n "${found}" ]; then cp "${found}" "/app/tools/tesseract/tessdata/${pack}"; fi; \
         fi; \
       done \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
