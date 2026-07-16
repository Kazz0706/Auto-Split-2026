# Raspberry Pi 5（Linux ARM64 / Python 3.11）と合わせた分割推論用イメージ
FROM --platform=linux/arm64 python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# OpenCVの実行に必要な共有ライブラリだけを導入する。
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY ./auto_split /app/auto_split

# Camera ModuleはRaspberry Pi OSのlibcamera/Picamera2と密結合のため、
# カメラ単体ベンチマークは現時点ではホストOS上で実行する。
CMD ["/bin/bash"]
