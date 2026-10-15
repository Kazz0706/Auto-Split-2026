"""Raspberry Pi native venvでSplitYOLOWrapper全体を測定する。

JPEG読込、分割推論と同じ前処理、全レイヤforward、NMSを個別に計測する。
YOLO.predict()は使わず、cloud_profile.pyと同一のモデル実行経路を用いる。
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics.utils.nms import non_max_suppression

from auto_split.src.split_model import SplitYOLOWrapper


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "yolov8n.pt"
DEFAULT_IMAGE = PROJECT_DIR / "samples" / "test.jpg"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_full_model(wrapper: SplitYOLOWrapper, img: torch.Tensor) -> torch.Tensor:
    """分割推論の接続規則を保ったまま、全レイヤを実行する。"""
    x = img
    saved: list[torch.Tensor | None] = []
    for layer in wrapper.layers:
        if layer.f == -1:
            layer_input = x
        elif isinstance(layer.f, int):
            layer_input = saved[layer.f]
        else:
            layer_input = [x if index == -1 else saved[index] for index in layer.f]
        x = layer(layer_input)
        saved.append(x if len(saved) in wrapper.model.save else None)
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def one_run(wrapper: SplitYOLOWrapper, image_path: Path) -> dict[str, float]:
    total_start = time.perf_counter()
    img, _, _ = wrapper.preprocess(image_path)
    preprocess_end = time.perf_counter()

    synchronize(wrapper.device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        prediction = run_full_model(wrapper, img)
    synchronize(wrapper.device)
    forward_end = time.perf_counter()

    nms_start = time.perf_counter()
    with torch.inference_mode():
        raw_prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        non_max_suppression(raw_prediction.detach().clone())
    synchronize(wrapper.device)
    nms_end = time.perf_counter()

    timings = wrapper.last_preprocess_timing
    return {
        "image_read_ms": timings.get("image_read_ms", 0.0),
        "preprocess_ms": timings["total_ms"],
        "forward_ms": (forward_end - forward_start) * 1000,
        "nms_ms": (nms_end - nms_start) * 1000,
        "end_to_end_ms": (nms_end - total_start) * 1000,
        "read_and_preprocess_wall_ms": (preprocess_end - total_start) * 1000,
    }


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("--runs は1以上、--warmup は0以上にしてください。")
    if not args.model.is_file() or not args.image.is_file():
        raise FileNotFoundError("--model または --image のファイルが見つかりません。")

    load_start = time.perf_counter()
    wrapper = SplitYOLOWrapper(args.model)
    synchronize(wrapper.device)
    load_ms = (time.perf_counter() - load_start) * 1000
    print(f"device={wrapper.device} image={args.image} warmup={args.warmup} runs={args.runs}")

    for _ in range(args.warmup):
        one_run(wrapper, args.image)

    records = [one_run(wrapper, args.image) for _ in range(args.runs)]
    print("\n===== Raspberry Pi SplitYOLOWrapper Profile =====")
    print(f"model_loading_ms: {load_ms:.3f}")
    for name in records[0]:
        values = [record[name] for record in records]
        print(
            f"{name}: mean={statistics.fmean(values):.3f} ms "
            f"p50={percentile(values, 0.50):.3f} ms "
            f"p95={percentile(values, 0.95):.3f} ms"
        )
    print(f"throughput_fps: {1000 / statistics.fmean([r['end_to_end_ms'] for r in records]):.3f}")


if __name__ == "__main__":
    main()
