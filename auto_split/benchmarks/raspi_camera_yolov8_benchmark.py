"""Raspberry Pi Camera Moduleから直接YOLOv8を実行する単体ベンチマーク。

JPEGファイルへ保存・再読込せず、Picamera2が取得したフレームをNumPy配列のまま
Ultralytics YOLOへ渡す。取得、色変換、YOLO処理を個別に計測する。
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "yolov8n.pt"


def percentile(values: list[float], fraction: float) -> float:
    """補間付きパーセンタイルを返す（値の単位はms）。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def synchronize_if_cuda(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=100, help="計測フレーム数")
    parser.add_argument("--warmup", type=int, default=10, help="ウォームアップ回数")
    parser.add_argument("--device", default="cpu", help='例: "cpu" または "0"')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--save",
        type=Path,
        help="指定時のみ最終フレームの検出結果を保存する（計測後に実行）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("--frames は1以上、--warmup は0以上にしてください。")
    if not args.model.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model}")

    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise SystemExit(
            "Picamera2が必要です。Raspberry Pi OS上で "
            "`sudo apt install -y python3-picamera2` を実行してください。"
        ) from exc

    model = YOLO(str(args.model))
    model.to(args.device)

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)  # 自動露出・ホワイトバランスを安定させる

    try:
        print(
            f"Camera: {args.width}x{args.height}, model: {args.model.name}, "
            f"device: {args.device}"
        )
        print(f"Warmup: {args.warmup}, measured frames: {args.frames}")

        # モデル初期化、メモリ確保、カメラの最初のフレームを計測対象から除外する。
        for _ in range(args.warmup):
            frame_rgb = camera.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _ = model.predict(
                source=frame_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            synchronize_if_cuda(args.device)

        capture_ms: list[float] = []
        conversion_ms: list[float] = []
        yolo_ms: list[float] = []
        total_ms: list[float] = []
        last_result = None

        for index in range(args.frames):
            total_start = time.perf_counter()

            capture_start = time.perf_counter()
            frame_rgb = camera.capture_array("main")
            capture_end = time.perf_counter()

            conversion_start = time.perf_counter()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            conversion_end = time.perf_counter()

            synchronize_if_cuda(args.device)
            yolo_start = time.perf_counter()
            results = model.predict(
                source=frame_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            synchronize_if_cuda(args.device)
            yolo_end = time.perf_counter()

            capture_ms.append((capture_end - capture_start) * 1000)
            conversion_ms.append((conversion_end - conversion_start) * 1000)
            yolo_ms.append((yolo_end - yolo_start) * 1000)
            total_ms.append((yolo_end - total_start) * 1000)
            last_result = results[0]

            print(
                f"frame={index + 1:03d} "
                f"capture={capture_ms[-1]:7.2f} ms "
                f"convert={conversion_ms[-1]:6.2f} ms "
                f"yolo={yolo_ms[-1]:7.2f} ms "
                f"total={total_ms[-1]:7.2f} ms"
            )

        print("\n===== Raspberry Pi Camera → YOLOv8 Result =====")
        for name, values in (
            ("Capture", capture_ms),
            ("RGB→BGR", conversion_ms),
            ("YOLO (preprocess + inference + NMS)", yolo_ms),
            ("End-to-end", total_ms),
        ):
            print(
                f"{name:36s} "
                f"mean={statistics.fmean(values):8.2f} ms "
                f"p50={percentile(values, 0.50):8.2f} ms "
                f"p95={percentile(values, 0.95):8.2f} ms"
            )
        print(f"End-to-end FPS: {1000 / statistics.fmean(total_ms):.2f}")

        if args.save and last_result is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), last_result.plot())
            print(f"Saved final detection image: {args.save}")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
