# Container images

This directory separates the deployable images by role.

* `edge/`: CPU-only Raspberry Pi split-inference client.  The image is shared
  by all Raspberry Pi nodes and is published as an ARM64 image.
* `cloud/`: split-inference server for the current MacBook CPU baseline.
  `Dockerfile.dgx` is the DGX Spark GPU variant.

Both Dockerfiles are built with the repository root as their build context, so
they can copy the shared `auto_split` package.

## Build and publish

Run these commands on the MacBook.  Both the Apple Silicon MacBook and
Raspberry Pi 5 use `linux/arm64`.

```sh
TAG=20260723-py311
docker login

docker buildx build --platform linux/arm64 \
  -f containers/cloud/Dockerfile \
  -t kazumayoshida/yolov8-autosplit-cloud:${TAG} \
  --push .

docker buildx build --platform linux/arm64 \
  -f containers/edge/Dockerfile \
  -t kazumayoshida/yolov8-autosplit-edge:${TAG} \
  --push .
```

## Run the current CPU baseline

`buildx --push` publishes the image but does not guarantee that it is loaded
into the MacBook's local image store.  Pull the just-published cloud image,
then start the server:

```sh
docker pull kazumayoshida/yolov8-autosplit-cloud:20260723-py311
docker run --rm -it --name autosplit-cloud -p 5001:5001 \
  kazumayoshida/yolov8-autosplit-cloud:20260723-py311
```

On the Raspberry Pi, pull and run the edge client with a static image:

```sh
docker pull kazumayoshida/yolov8-autosplit-edge:20260723-py311
docker run --rm -it \
  -e CLOUD_HOST=192.168.12.30 \
  -v "$(pwd)/auto_split/outputs:/app/auto_split/outputs" \
  kazumayoshida/yolov8-autosplit-edge:20260723-py311 \
  --source image --frames 100 --no-save
```

`Picamera2` is not listed in the edge image requirements.  It is supplied by
Raspberry Pi OS together with a matching libcamera Python binding, rather than
as an ordinary portable pip dependency.  Until a Pi-OS-specific camera base
image is validated, run camera capture on the Pi host, or use `--source image`
inside this container.

The corresponding Pi-host venv specification is in `../edge/`.


# Pi Camera Raw + Pi + DGX のsplit=3実行
[DGX Spark]
docker run --rm -it \
  --name autosplit-cloud-dgx \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5001:5001 \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -v "$(pwd)/auto_split:/app/auto_split" \
  yolov8-autosplit-cloud:dgx-gpu-20260727

[Raspberry Pi]
python -m auto_split.src.edge \
  --source camera \
  --cloud-host <DGXのLAN_IP> \
  --cloud-port 5001 \
  --split 3 \
  --frames 100 \
  --camera-width 1280 \
  --camera-height 720 \
  --no-save