# Raspberry Pi native camera edge

This directory defines the **host-native** environment used for Camera Module
3 NoIR capture and the edge half of split inference.  It is separate from the
portable container environment in `../containers/edge/`.

Picamera2 and libcamera are installed from Raspberry Pi OS, not from pip.  The
venv is created with `--system-site-packages` so it can use that OS-managed
camera binding while the ML packages remain isolated in the venv.

Python 3.13 (Pi OS Trixie) requires the PyYAML version pinned in this
directory; do not substitute the older container-only requirements file.

## One-time Raspberry Pi setup

```sh
cd ~/auto_split_2026

sudo apt update
sudo apt install -y python3-venv python3-picamera2 libgl1 libglib2.0-0

python3 -m venv --system-site-packages edge/.venv
source edge/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r edge/requirements.txt

python -c "from picamera2 import Picamera2; import torch, ultralytics; print('camera and ML environment ready')"
```

## Run camera split inference

Start `containers/cloud` on the MacBook first.  Then, on the Raspberry Pi:

```sh
cd ~/auto_split_2026
source edge/.venv/bin/activate

CLOUD_HOST=192.168.12.29 \
python -m auto_split.src.edge \
  --source camera \
  --frames 100 \
  --camera-width 1280 \
  --camera-height 720 \
  --output-dir auto_split/outputs/camera_split \
  --output-prefix camera_split
```

For throughput-only measurement, add `--no-save`.  Without it, one annotated
JPEG is written for every inferred frame.

Leave the venv with `deactivate`.
