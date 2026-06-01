# raspi_yolov8_benchmark.py
# Raspberry Pi上でYOLOv8単体推論速度を測定するコード

import time
import torch
from ultralytics import YOLO

# -------------------------
# 設定
# -------------------------
MODEL_PATH = "yolov8n.pt"
IMAGE_PATH = "images/test.jpg"

WARMUP = 5      # ウォームアップ回数
RUNS = 30       # 計測回数

# -------------------------
# モデル読み込み
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = YOLO(MODEL_PATH)
model.to(device)

print("Device:", device)

# -------------------------
# ウォームアップ
# -------------------------
print(f"Warmup {WARMUP} runs...")

for _ in range(WARMUP):
    _ = model.predict(
        source=IMAGE_PATH,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device=device
    )

# -------------------------
# 本計測
# -------------------------
print(f"Benchmark {RUNS} runs...")

times = []

for i in range(RUNS):

    start = time.perf_counter()

    _ = model.predict(
        source=IMAGE_PATH,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device=device
    )

    end = time.perf_counter()

    t = end - start
    times.append(t)

    print(f"Run {i+1:02d}: {t*1000:.2f} ms")

# -------------------------
# 結果
# -------------------------
avg = sum(times) / len(times)
fps = 1.0 / avg

print("\n===== RESULT =====")
print(f"Average Latency : {avg*1000:.2f} ms")
print(f"FPS             : {fps:.2f}")
print(f"Min Latency     : {min(times)*1000:.2f} ms")
print(f"Max Latency     : {max(times)*1000:.2f} ms")