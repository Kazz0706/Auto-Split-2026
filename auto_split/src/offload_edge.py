"""Pi CameraフレームをRawまたはJPEGとしてDGXへ全オフロードするクライアント。"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from auto_split.src.protocol import recv_message, send_message


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_DIR / "samples" / "test.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "offload"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camera", "image"), default="camera")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--transport", choices=("raw", "jpeg"), default="raw")
    parser.add_argument(
        "--raw-quantization",
        choices=("off", "int8"),
        default="off",
        help="Raw BGRの近似INT8化。uint8 Rawと同じ1 byte/pixelで、通信量は減らない。",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--cloud-host", default=os.environ.get("CLOUD_HOST"))
    parser.add_argument("--cloud-port", type=int, default=int(os.environ.get("CLOUD_PORT", "5002")))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10, help="DGX推論を温める送信回数")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="offload")
    parser.add_argument("--save-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


class CameraFrameSource:
    """Picamera2を一度だけ起動し、連続してBGRフレームを返す。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera = None

    def __enter__(self) -> "CameraFrameSource":
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise SystemExit(
                "Picamera2が必要です。Raspberry Pi OSで "
                "`sudo apt install -y python3-picamera2` を実行してください。"
            ) from exc
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            # Picamera2のBGR888はNumPy上ではRGB順。下のRGB→BGR変換で
            # OpenCV/YOLOが期待するBGRにする。
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


def encode_frame(frame_bgr: np.ndarray, args: argparse.Namespace, frame_id: int) -> tuple[dict[str, Any], bytes, dict[str, float]]:
    """RawまたはJPEGのpayloadを作り、局所処理時間を返す。"""
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("送信フレームは形状(H, W, 3)のuint8 BGR配列である必要があります。")
    frame_bgr = np.ascontiguousarray(frame_bgr)
    metrics = {"raw_quantization_ms": 0.0, "jpeg_encode_ms": 0.0}

    if args.transport == "jpeg":
        start = time.perf_counter()
        ok, encoded = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        )
        metrics["jpeg_encode_ms"] = (time.perf_counter() - start) * 1000
        if not ok:
            raise RuntimeError("JPEGエンコードに失敗しました。")
        return {
            "kind": "frame",
            "transport": "jpeg",
            "frame_id": frame_id,
            "jpeg_quality": args.jpeg_quality,
        }, encoded.tobytes(), metrics

    header: dict[str, Any] = {
        "kind": "frame",
        "transport": "raw",
        "frame_id": frame_id,
        "shape": list(frame_bgr.shape),
        "raw_quantization": args.raw_quantization,
    }
    if args.raw_quantization == "off":
        header["dtype"] = "uint8"
        return header, frame_bgr.tobytes(), metrics

    # Camera Rawは元からuint8であり、INT8化しても1 byte/pixelのままである。
    # 中心128・動的scaleによる近似化を、通信量ではなく精度/処理時間用に比較する。
    start = time.perf_counter()
    centered = frame_bgr.astype(np.float32) - 128.0
    scale = max(float(np.abs(centered).max()) / 127.0, 1e-8)
    quantized = np.clip(np.rint(centered / scale), -127, 127).astype(np.int8)
    metrics["raw_quantization_ms"] = (time.perf_counter() - start) * 1000
    header.update({"dtype": "int8", "scale": scale, "zero_point": 128.0})
    return header, quantized.tobytes(), metrics


def request_cloud(host: str, port: int, header: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], dict[str, float]]:
    """1フレームを送信して応答を待つ。既存split実験と同じく接続は毎回作る。"""
    total_start = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connect_start = time.perf_counter()
        sock.connect((host, port))
        connect_end = time.perf_counter()
        send_start = time.perf_counter()
        request_wire_bytes = send_message(sock, header, payload)
        send_end = time.perf_counter()
        response_header, _, response_wire_bytes = recv_message(sock)
        response_end = time.perf_counter()
    if response_header.get("kind") == "error":
        raise RuntimeError(f"DGX側エラー: {response_header.get('message', '詳細なし')}")
    if response_header.get("kind") != "result":
        raise RuntimeError("DGX側から未知の応答を受信しました。")
    return response_header, {
        "tcp_connect_ms": (connect_end - connect_start) * 1000,
        "request_send_ms": (send_end - send_start) * 1000,
        "communication_and_cloud_roundtrip_ms": (response_end - total_start) * 1000,
        "request_wire_bytes": float(request_wire_bytes),
        "response_wire_bytes": float(response_wire_bytes),
    }


def save_result(frame_bgr: np.ndarray, response: dict[str, Any], path: Path) -> float:
    start = time.perf_counter()
    boxes = response.get("boxes", [])
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 6), dtype=torch.float32)
    names = {int(key): value for key, value in response["names"].items()}
    result = Results(orig_img=frame_bgr, path="frame", names=names, boxes=boxes_tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), result.plot()):
        raise RuntimeError(f"検出結果を保存できません: {path}")
    return (time.perf_counter() - start) * 1000


def profile_frame(frame_bgr: np.ndarray, source_timing: dict[str, float], args: argparse.Namespace, frame_id: int, save: bool) -> dict[str, float]:
    total_start = time.perf_counter()
    header, payload, encode_timing = encode_frame(frame_bgr, args, frame_id)
    encode_end = time.perf_counter()
    response, request_timing = request_cloud(args.cloud_host, args.cloud_port, header, payload)
    response_end = time.perf_counter()

    plot_ms = 0.0
    if save:
        path = args.output_dir / f"{args.output_prefix}_{frame_id:06d}.{args.save_format}"
        plot_ms = save_result(frame_bgr, response, path)

    server_timing = response["timings"]
    camera_ms = source_timing.get("camera_capture_ms", 0.0) + source_timing.get("rgb_to_bgr_ms", 0.0)
    inference_ms = camera_ms + (response_end - total_start) * 1000
    metrics: dict[str, float] = {
        **source_timing,
        **encode_timing,
        "packet_build_ms": (encode_end - total_start) * 1000 - encode_timing["raw_quantization_ms"] - encode_timing["jpeg_encode_ms"],
        **request_timing,
        "server_payload_restore_ms": float(server_timing["payload_restore_ms"]),
        "server_raw_dequantization_ms": float(server_timing["raw_dequantization_ms"]),
        "server_yolo_preprocess_ms": float(server_timing["yolo_preprocess_ms"]),
        "server_yolo_forward_ms": float(server_timing["yolo_forward_ms"]),
        "server_yolo_postprocess_nms_ms": float(server_timing["yolo_postprocess_nms_ms"]),
        "server_yolo_predict_wall_ms": float(server_timing["yolo_predict_wall_ms"]),
        "server_cloud_processing_ms": float(server_timing["cloud_processing_ms"]),
        "result_plot_ms": plot_ms,
        "end_to_end_ms_without_plot": inference_ms,
        "end_to_end_ms_with_plot": inference_ms + plot_ms,
    }
    print(
        f"frame={frame_id:06d} transport={args.transport} "
        f"request_kb={metrics['request_wire_bytes'] / 1024:.2f} "
        f"response_kb={metrics['response_wire_bytes'] / 1024:.2f} "
        f"end_to_end_ms={metrics['end_to_end_ms_without_plot']:.3f}"
    )
    return metrics


def print_summary(records: list[dict[str, float]]) -> None:
    print("\n===== Edge → DGX Full Offload Summary =====")
    for name in records[0]:
        values = [record[name] for record in records]
        unit = "KB" if name.endswith("_bytes") else "ms"
        if unit == "KB":
            values = [value / 1024 for value in values]
        print(
            f"{name}: mean={statistics.fmean(values):.3f} {unit} "
            f"p50={percentile(values, 0.50):.3f} {unit} "
            f"p95={percentile(values, 0.95):.3f} {unit}"
        )
    fps_without_plot = 1000 / statistics.fmean([r["end_to_end_ms_without_plot"] for r in records])
    print(f"throughput_fps: {fps_without_plot:.3f}")
    print(f"throughput_fps_without_plot: {fps_without_plot:.3f}")
    print(f"throughput_fps_with_plot: {1000 / statistics.fmean([r['end_to_end_ms_with_plot'] for r in records]):.3f}")


def main() -> None:
    args = parse_args()
    if not args.cloud_host:
        raise SystemExit("--cloud-host または環境変数 CLOUD_HOST を指定してください。")
    if args.frames < 1 or args.warmup < 0 or args.camera_warmup < 0:
        raise SystemExit("--frames は1以上、--warmup と --camera-warmup は0以上にしてください。")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality は1〜100で指定してください。")

    records: list[dict[str, float]] = []
    if args.source == "camera":
        with CameraFrameSource(args) as camera:
            print(f"Warmup requests: {args.warmup}, measured frames: {args.frames}")
            for frame_id in range(1, args.warmup + 1):
                frame, source_timing = camera.capture()
                profile_frame(frame, source_timing, args, frame_id, save=False)
            for frame_id in range(1, args.frames + 1):
                frame, source_timing = camera.capture()
                records.append(profile_frame(frame, source_timing, args, frame_id, save=not args.no_save))
    else:
        read_start = time.perf_counter()
        image = cv2.imread(str(args.image))
        image_read_ms = (time.perf_counter() - read_start) * 1000
        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {args.image}")
        for frame_id in range(1, args.warmup + 1):
            profile_frame(image.copy(), {"image_read_ms": image_read_ms}, args, frame_id, save=False)
        for frame_id in range(1, args.frames + 1):
            records.append(profile_frame(image.copy(), {"image_read_ms": image_read_ms}, args, frame_id, save=not args.no_save))

    print_summary(records)


if __name__ == "__main__":
    main()
