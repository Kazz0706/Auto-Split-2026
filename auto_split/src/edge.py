"""エッジ側のYOLOv8分割推論クライアント。"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import socket
import statistics
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from auto_split.src.split_model import SplitYOLOWrapper


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"
IMAGE_PATH = PROJECT_DIR / "samples" / "test.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("image", "camera"), default="image")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--split", type=int, default=3)
    parser.add_argument("--cloud-host", default=os.environ.get("CLOUD_HOST"))
    parser.add_argument("--cloud-port", type=int, default=int(os.environ.get("CLOUD_PORT", "5001")))
    parser.add_argument("--frames", type=int, default=1, help="処理するフレーム数")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="split")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


class CameraFrameSource:
    """Picamera2を一度だけ起動し、連続してBGRフレームを取得する。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera = None

    def __enter__(self) -> "CameraFrameSource":
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise SystemExit(
                "Picamera2が必要です。Raspberry Pi OS上で "
                "`sudo apt install -y python3-picamera2` を実行してください。"
            ) from exc

        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            # Picamera2のBGR888はNumPy上ではRGB順。下でOpenCV用BGRへ変換する。
            main={"size": (self.args.camera_width, self.args.camera_height), "format": "BGR888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1.0)
        for _ in range(self.args.camera_warmup):
            self.camera.capture_array("main")
        return self

    def capture(self) -> tuple[np.ndarray, dict[str, float]]:
        capture_start = time.perf_counter()
        frame_rgb = self.camera.capture_array("main")
        capture_end = time.perf_counter()

        conversion_start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        conversion_end = time.perf_counter()
        return frame_bgr, {
            "camera_capture_ms": (capture_end - capture_start) * 1000,
            "rgb_to_bgr_ms": (conversion_end - conversion_start) * 1000,
        }

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.camera is not None:
            self.camera.stop()


def recvall(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("クラウド側が応答の途中で接続を閉じました。")
        chunks.extend(chunk)
    return bytes(chunks)


def build_packet(wrapper, edge_out, context, meta, split_point) -> tuple[bytes, float]:
    start = time.perf_counter()
    scale_q = edge_out.abs().max().clamp_min(1e-8) / 127
    edge_out_q = torch.round(edge_out / scale_q).to(torch.int8)
    tensor_buffer = io.BytesIO()
    torch.save(edge_out_q.cpu(), tensor_buffer)

    context_indices, context_tensors, context_scales = [], [], []
    for index in sorted(wrapper.get_needed_context(split_point)):
        tensor = context[index]
        if tensor is None:
            continue
        tensor_cpu = tensor.cpu()
        scale = tensor_cpu.abs().max().clamp_min(1e-8) / 127
        context_indices.append(index)
        context_scales.append(float(scale.item()))
        context_tensors.append(torch.round(tensor_cpu / scale).to(torch.int8))

    context_buffer = io.BytesIO()
    torch.save(context_tensors, context_buffer)
    packet = {
        "tensor": tensor_buffer.getvalue(),
        "context": context_buffer.getvalue(),
        "context_idx": np.array(context_indices, dtype=np.int32),
        "meta": json.dumps(meta).encode(),
        "scale_q": float(scale_q.item()),
        "context_scales": np.array(context_scales, dtype=np.float32),
        "split": split_point,
    }
    return pickle.dumps(packet), (time.perf_counter() - start) * 1000


def request_cloud(host: str, port: int, packet: bytes) -> tuple[dict, float]:
    start = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.sendall(struct.pack(">I", len(packet)) + packet)
        response_size = struct.unpack(">I", recvall(sock, 4))[0]
        response = recvall(sock, response_size)
    return pickle.loads(response), (time.perf_counter() - start) * 1000


def save_result(orig_img: np.ndarray, boxes: list, names: dict, path: Path) -> float:
    start = time.perf_counter()
    # Ultralytics expects detections in an (N, 6) or (N, 7) tensor.  An empty
    # Python list becomes a 1-D tensor with shape (0,), so preserve the box
    # width explicitly when the cloud found no objects.
    boxes_tensor = (
        torch.tensor(boxes, dtype=torch.float32)
        if boxes
        else torch.empty((0, 6), dtype=torch.float32)
    )
    result = Results(orig_img=orig_img, path="frame", names=names, boxes=boxes_tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), result.plot())
    return (time.perf_counter() - start) * 1000


def profile_frame(wrapper, args, frame_index, orig_img, source_timing) -> dict[str, float]:
    preprocess_start = time.perf_counter()
    img, orig_img, meta = wrapper.preprocess_frame(orig_img)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    edge_start = time.perf_counter()
    with torch.inference_mode():
        edge_out, context = wrapper.run_edge(img, args.split)
    edge_ms = (time.perf_counter() - edge_start) * 1000

    packet, compression_ms = build_packet(wrapper, edge_out, context, meta, args.split)
    result_packet, roundtrip_ms = request_cloud(args.cloud_host, args.cloud_port, packet)

    plot_ms = 0.0
    if not args.no_save:
        output_path = args.output_dir / f"{args.output_prefix}_{frame_index:06d}.jpg"
        plot_ms = save_result(orig_img, result_packet["boxes"], wrapper.yolo.names, output_path)

    metrics = {
        **source_timing,
        **{f"preprocess_{key}": value for key, value in wrapper.last_preprocess_timing.items()},
        "preprocess_total_ms": preprocess_ms,
        "edge_inference_ms": edge_ms,
        "compression_and_serialization_ms": compression_ms,
        "communication_and_cloud_roundtrip_ms": roundtrip_ms,
        "cloud_processing_ms": result_packet["cloud_time"] * 1000,
        "result_plot_ms": plot_ms,
        "end_to_end_ms_without_plot": sum(source_timing.values()) + preprocess_ms + edge_ms + compression_ms + roundtrip_ms,
    }
    print(
        f"frame={frame_index:06d} size_kb={(4 + len(packet)) / 1024:.2f} "
        f"end_to_end_ms={metrics['end_to_end_ms_without_plot']:.3f}"
    )
    return metrics


def print_summary(records: list[dict[str, float]]) -> None:
    print("\n===== Edge → Cloud Split Inference Summary =====")
    for name in records[0]:
        values = [record[name] for record in records]
        print(f"{name}: mean={statistics.fmean(values):.3f} ms")
    print(f"throughput_fps: {1000 / statistics.fmean([r['end_to_end_ms_without_plot'] for r in records]):.3f}")


def main() -> None:
    args = parse_args()
    if not args.cloud_host:
        raise SystemExit("--cloud-host または環境変数 CLOUD_HOST を指定してください。")
    if args.frames < 1:
        raise SystemExit("--frames は1以上にしてください。")
    if not args.model.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model}")

    load_start = time.perf_counter()
    wrapper = SplitYOLOWrapper(args.model)
    print(f"model_loading_ms: {(time.perf_counter() - load_start) * 1000:.3f}")

    records = []
    if args.source == "camera":
        with CameraFrameSource(args) as camera:
            for frame_index in range(1, args.frames + 1):
                frame, source_timing = camera.capture()
                records.append(profile_frame(wrapper, args, frame_index, frame, source_timing))
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {args.image}")
        for frame_index in range(1, args.frames + 1):
            records.append(profile_frame(wrapper, args, frame_index, image.copy(), {}))

    print_summary(records)


if __name__ == "__main__":
    main()
