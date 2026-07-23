# 実行構成

## 主要プログラム

- `src/edge.py`: Raspberry Piなどのエッジ側で、前処理・前半推論・特徴量送信を行う。
- `src/cloud.py`: サーバー側で、特徴量受信・後半推論・検出結果の返送を行う。
- `src/split_model.py`: エッジ・サーバーで共有するYOLOv8分割推論ロジック。

プロジェクトのルートから、モジュールとして実行する。

```bash
python -m auto_split.src.cloud
python -m auto_split.src.edge
```

## データと実験

- `models/`: 学習済み重み。現在は `yolov8n.pt`。
- `samples/`: 再現実験用の静止画入力。
- `benchmarks/`: 分割なし・層別プロファイルなどの比較実験。
- `tools/`: 分割点探索、ONNX出力、モデル構造解析。
- `outputs/`: 実行時生成物。Git管理しない。

ベンチマークと解析もプロジェクトのルートからモジュールとして実行する。

```bash
python -m auto_split.benchmarks.normal_yolov8
python -m auto_split.benchmarks.raspi_yolov8_benchmark
python -m auto_split.tools.search
```

## Docker（分割推論実行用）

`Dockerfile` はPython 3.11、PyTorch 2.9.0、Torchvision 0.24.0、
Ultralytics 8.3.222を `requirements-runtime.txt` で固定している。
OpenCV 4.12.0.88の新規導入要件に合わせ、DockerのNumPyは2.2.6を使用する。
TensorFlow、音声処理、ONNX解析は実行イメージから除外し、必要な場合だけ
`requirements-tools.txt` を利用する。

ビルドに成功した時点の推移依存も固定する場合は、Pi上で次を実行して
`requirements-runtime.lock` を保存する。

```bash
docker run --rm yolov8-autosplit:pi311 python -m pip freeze \
  > requirements-runtime.lock
```

```bash
docker build --platform linux/arm64 -t yolov8-autosplit:pi311 .
```

Mac/DGX上でクラウド側を起動し、Pi側ではクラウドのLAN内IPを
`CLOUD_HOST` として渡す。ポートは既定で5001であり、必要なら両側に
同じ `CLOUD_PORT` を指定する。

```bash
# Mac/DGX
docker run --rm -p 5001:5001 yolov8-autosplit:pi311 \
  python -m auto_split.src.cloud

# Raspberry Pi
docker run --rm \
  -e CLOUD_HOST=192.168.1.20 \
  -v "$(pwd)/auto_split/outputs:/app/auto_split/outputs" \
  yolov8-autosplit:pi311 \
  python -m auto_split.src.edge
```

`192.168.1.20` は実際のMac/DGXのIPアドレスへ置き換える。

Picamera2/libcameraはRaspberry Pi OSのシステムPythonと結び付くため、
カメラ単体ベンチマークはホストOSで実行する。カメラをコンテナ化するのは、
分割推論の基準性能を取得した後に、デバイス公開とPi OS依存を含む別構成として扱う。

### Raspberry Pi Camera Module 3 NoIRによる単体実行

`raspi_camera_yolov8_benchmark.py` は、Picamera2から取得したフレームを
JPEGへ保存せずNumPy配列のままYOLOv8へ入力する。Raspberry Pi OS上では
Picamera2をOSパッケージとして導入しておく。

```bash
sudo apt install -y python3-picamera2
python -m auto_split.benchmarks.raspi_camera_yolov8_benchmark --frames 100 --warmup 10
```

最終フレームの検出結果だけを保存する場合は、次を使う。

```bash
python -m auto_split.benchmarks.raspi_camera_yolov8_benchmark \
  --save auto_split/outputs/camera_detection.jpg
```
