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

### カメラを用いる分割推論

`edge.py` は、静止画（既定値）とPicamera2のフレームを同一の
`SplitYOLOWrapper.preprocess_frame()` へ渡す。カメラモードでは、JPEGへの保存・
再読込を行わず、カメラ取得、RGB→BGR変換、前処理、前半推論、量子化・
シリアライズ、通信・クラウド往復を個別に記録する。

まずクラウド側を起動してから、Raspberry Pi OS上で実行する。

```bash
python -m auto_split.src.edge \
  --source camera \
  --cloud-host 192.168.12.30 \
  --frames 100 \
  --camera-width 1280 \
  --camera-height 720 \
  --output-dir auto_split/outputs/camera_split \
  --output-prefix camera_split
```

`--frames 100` の場合、`camera_split_000001.jpg` から
`camera_split_000100.jpg` までを保存する。エッジは1フレームごとに
「取得→前処理→前半推論→クラウド往復→結果保存」を完了してから次のフレームを
取得する同期方式である。処理中にカメラ側で生成された古いフレームは利用しないため、
低遅延の基準測定に適している。将来のスループット最適化では、別途パイプライン化する。

このカメラモードには、Pi OSの `python3-picamera2` とlibcameraが必要である。
現在の軽量Dockerイメージにはこれらを含めないため、まずホストOSで分割推論の
カメラ入力を検証する。Docker内でのカメラ利用は、Pi OS/libcamera専用の
エッジイメージとデバイス公開を追加する次段階で扱う。

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

### Pi CameraからDGXへの全オフロード比較

`offload_edge.py` と `offload_cloud.py` は、既存の分割推論とは別に、
DGXで融合済みの `YOLO.predict()` 全体を実行する比較用プログラムである。
`raw` はBGR uint8、`jpeg` はPi上でエンコードしたJPEGを送る。

DGXではGPUコンテナを起動し、ポート5002を公開する。

```bash
docker run --rm -it \
  --name autosplit-offload-dgx \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5002:5002 \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -v "$(pwd)/auto_split:/app/auto_split" \
  yolov8-autosplit-cloud:dgx-gpu-20260727 \
  python3 -m auto_split.src.offload_cloud --port 5002 --device 0
```

Piではnative venvを有効化して、Raw全送信を実行する。

```bash
source edge/.venv/bin/activate
python -m auto_split.src.offload_edge \
  --source camera \
  --transport raw \
  --raw-quantization off \
  --cloud-host <DGXのLAN_IP> \
  --cloud-port 5002 \
  --warmup 10 \
  --frames 100 \
  --no-save
```

JPEG全送信は、`--transport jpeg --jpeg-quality 95` に変更する。
Rawフレームはもともとuint8のため、`--raw-quantization int8` は通信量を減らさず、
近似量子化の処理時間・検出結果への影響だけを比較する。

検出結果をPiへ保存する場合は`--no-save`を外し、必要に応じて出力先を指定する。

```bash
--output-dir auto_split/outputs/offload_raw --output-prefix raw --save-format png
```
