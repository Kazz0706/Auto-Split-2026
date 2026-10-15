"""Pi Camera入力YOLOv8単体推論の詳細内訳ベンチマーク。

元の raspi_camera_yolov8_benchmark.py は変更しない。この版はUltralyticsが
返す結果の speed 辞書を表示し、YOLO処理を前処理・forward・後処理へ分解する。
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
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu", help='例: "cpu" または "0"')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", type=Path, help="指定時のみ最終結果を保存する。")
    return parser.parse_args()


def print_metric(name: str, values: list[float]) -> None:
    print(
        f"{name}: mean={statistics.fmean(values):.3f} ms "
        f"p50={percentile(values, 0.50):.3f} ms "
        f"p95={percentile(values, 0.95):.3f} ms"
    )


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.warmup < 0:
        raise ValueError("--frames は1以上、--warmup は0以上にしてください。")
    if not args.model.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model}")

    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise SystemExit("Picamera2が必要です。`sudo apt install -y python3-picamera2` を実行してください。") from exc

    load_start = time.perf_counter()
    model = YOLO(str(args.model))
    model.to(args.device)
    synchronize_if_cuda(args.device)
    model_loading_ms = (time.perf_counter() - load_start) * 1000

    camera = Picamera2()
    config = camera.create_video_configuration(
        # Picamera2のBGR888はNumPy上ではRGB順。下でOpenCV用BGRへ変換する。
        main={"size": (args.width, args.height), "format": "BGR888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)

    try:
        print(
            f"device={args.device} camera={args.width}x{args.height} "
            f"model={args.model.name} warmup={args.warmup} frames={args.frames}"
        )
        print("image_read_ms: N/A (camera frame is passed directly; no JPEG read)")

        for _ in range(args.warmup):
            frame_rgb = camera.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _ = model.predict(source=frame_bgr, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            synchronize_if_cuda(args.device)

        records: dict[str, list[float]] = {
            "camera_capture_ms": [],
            "rgb_to_bgr_ms": [],
            "yolo_preprocess_ms": [],
            "yolo_forward_ms": [],
            "yolo_postprocess_nms_ms": [],
            "yolo_predict_wall_ms": [],
            "end_to_end_ms": [],
        }
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
            predict_start = time.perf_counter()
            results = model.predict(
                source=frame_bgr, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
            )
            synchronize_if_cuda(args.device)
            predict_end = time.perf_counter()

            result = results[0]
            speed = result.speed
            records["camera_capture_ms"].append((capture_end - capture_start) * 1000)
            records["rgb_to_bgr_ms"].append((conversion_end - conversion_start) * 1000)
            records["yolo_preprocess_ms"].append(float(speed.get("preprocess", 0.0)))
            records["yolo_forward_ms"].append(float(speed.get("inference", 0.0)))
            records["yolo_postprocess_nms_ms"].append(float(speed.get("postprocess", 0.0)))
            records["yolo_predict_wall_ms"].append((predict_end - predict_start) * 1000)
            records["end_to_end_ms"].append((predict_end - total_start) * 1000)
            last_result = result

            print(
                f"frame={index + 1:03d} capture={records['camera_capture_ms'][-1]:.2f} "
                f"convert={records['rgb_to_bgr_ms'][-1]:.2f} "
                f"pre={records['yolo_preprocess_ms'][-1]:.2f} "
                f"forward={records['yolo_forward_ms'][-1]:.2f} "
                f"nms={records['yolo_postprocess_nms_ms'][-1]:.2f} "
                f"e2e={records['end_to_end_ms'][-1]:.2f} ms"
            )

        print("\n===== Raspberry Pi Camera Detailed Profile =====")
        print(f"model_loading_ms: {model_loading_ms:.3f}")
        for name, values in records.items():
            print_metric(name, values)
        print(f"throughput_fps: {1000 / statistics.fmean(records['end_to_end_ms']):.3f}")

        if args.save and last_result is not None:
            save_start = time.perf_counter()
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), last_result.plot())
            print(f"result_plot_and_save_ms: {(time.perf_counter() - save_start) * 1000:.3f}")
            print(f"saved: {args.save}")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
