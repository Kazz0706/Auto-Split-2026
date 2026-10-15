# Cloud / edge-server image for YOLOv8 split inference.
# Build from the repository root:
#   docker buildx build -f containers/cloud/Dockerfile ... .
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

COPY containers/cloud/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY auto_split /app/auto_split

EXPOSE 5001
CMD ["python", "-m", "auto_split.src.cloud"]
