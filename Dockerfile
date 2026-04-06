# Python3.11対応, 2026年度版Auto-Split
# ARG TARGETPLATFORM=linux/amd64
# FROM --platform=$TARGETPLATFORM python:3.11-slim
FROM --platform=linux/arm64 python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 基本ツール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    nano \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# python:3.11-slimは超軽量なのでGUI系、GL系、threading系が削除されている

# pip 更新
RUN pip install --upgrade pip

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY requirements-ml2.txt .
RUN pip install --no-cache-dir -r requirements-ml2.txt

COPY ./auto_split /app/auto_split

# デフォルト shell
CMD ["/bin/bash"]


# volume mount(Macでの編集をコンテナに反映)出来ないので、プログラムが完成したら解放
# COPY ./auto_split /app/auto_split
# COPY ./requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt

# 流れ: 同一ディレクトリ内でdocker build(イメージ作成) -> docker run(コンテナ起動)
# docker build -t yolov8-autosplit:py311 .
# docker build --platform linux/arm64 -t yolov8-autosplit:py311 .

# docker run --rm -it \
#     -v $(pwd)/auto_split:/app/auto_split \
#     -v $(pwd)/requirements.txt:/app/requirements.txt \
#     yolov8-autosplit:py311 \
#     /bin/bash

# yolov8-containerというコンテナ名にして保存したい場合(rmしない)
# docker run -it \
#     --name yolov8-container \
#     -v $(pwd)/auto_split:/app/auto_split \
#     -v $(pwd)/requirements.txt:/app/requirements.txt \
#     <yourname>/yolov8-autosplit:py311 \
#     /bin/bash