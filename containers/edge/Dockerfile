# Raspberry Pi edge inference image (Linux ARM64 / Python 3.11).
# It supports `--source image` immediately.  Camera capture is intentionally
# not installed here: Picamera2/libcamera must match the Pi OS system Python
# and libcamera stack.  Use a Pi-OS-specific camera base before enabling
# `--source camera` inside Docker.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

COPY containers/edge/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY auto_split /app/auto_split

CMD ["python", "-m", "auto_split.src.edge"]
