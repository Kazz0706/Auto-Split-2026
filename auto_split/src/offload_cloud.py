"""Raw/JPEG全オフロード用のDGXサーバー。融合済みYOLO.predict()で全推論する。"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from auto_split.src.protocol import recv_message, send_message


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLOUD_PORT", "5002")))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0", help='DGX GPUは"0"、CPU比較は"cpu"')
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def restore_raw(header: dict[str, Any], payload: bytes) -> tuple[np.ndarray, float]:
    shape = header.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("Raw frameのshapeが不正です。")
    expected_elements = int(np.prod(shape))
    quantization = header.get("raw_quantization")
    if quantization == "off":
        if header.get("dtype") != "uint8" or len(payload) != expected_elements:
            raise ValueError("uint8 Raw frameのサイズまたはdtypeが不正です。")
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape), 0.0
    if quantization != "int8" or header.get("dtype") != "int8" or len(payload) != expected_elements:
        raise ValueError("INT8 Raw frameのサイズ、dtype、または量子化指定が不正です。")

    scale = float(header.get("scale", 0.0))
    zero_point = float(header.get("zero_point", 128.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("INT8 Raw frameのscaleが不正です。")
    start = time.perf_counter()
    quantized = np.frombuffer(payload, dtype=np.int8).reshape(shape)
    frame = np.clip(np.rint(quantized.astype(np.float32) * scale + zero_point), 0, 255).astype(np.uint8)
    return frame, (time.perf_counter() - start) * 1000


def restore_frame(header: dict[str, Any], payload: bytes) -> tuple[np.ndarray, dict[str, float]]:
    """通信payloadをBGR uint8フレームへ復元する。"""
    restore_start = time.perf_counter()
    transport = header.get("transport")
    if transport == "raw":
        frame, dequantization_ms = restore_raw(header, payload)
    elif transport == "jpeg":
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        dequantization_ms = 0.0
        if frame is None:
            raise ValueError("JPEGデコードに失敗しました。")
    else:
        raise ValueError(f"未対応のtransportです: {transport!r}")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("復元結果は形状(H, W, 3)のuint8 BGRである必要があります。")
    return frame, {
        "payload_restore_ms": (time.perf_counter() - restore_start) * 1000,
        "raw_dequantization_ms": dequantization_ms,
    }


def process_frame(header: dict[str, Any], payload: bytes, model: YOLO, args: argparse.Namespace) -> dict[str, Any]:
    cloud_start = time.perf_counter()
    frame_bgr, timings = restore_frame(header, payload)

    synchronize_if_cuda()
    predict_start = time.perf_counter()
    results = model.predict(
        source=frame_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    synchronize_if_cuda()
    predict_end = time.perf_counter()

    result = results[0]
    speed = result.speed
    timings.update(
        {
            "yolo_preprocess_ms": float(speed.get("preprocess", 0.0)),
            "yolo_forward_ms": float(speed.get("inference", 0.0)),
            "yolo_postprocess_nms_ms": float(speed.get("postprocess", 0.0)),
            "yolo_predict_wall_ms": (predict_end - predict_start) * 1000,
            "cloud_processing_ms": (predict_end - cloud_start) * 1000,
        }
    )
    boxes = result.boxes.data.detach().cpu().tolist() if result.boxes is not None else []
    return {
        "kind": "result",
        "frame_id": header.get("frame_id"),
        "boxes": boxes,
        "names": {str(key): value for key, value in model.names.items()},
        "timings": timings,
    }


def send_error(connection: socket.socket, message: str) -> None:
    send_message(connection, {"kind": "error", "message": message})


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model}")
    load_start = time.perf_counter()
    model = YOLO(str(args.model))
    load_ms = (time.perf_counter() - load_start) * 1000
    print(f"model_loading_ms={load_ms:.3f} device_request={args.device}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen()
        print(f"Full-offload cloud server listening on {args.host}:{args.port}. Stop with Ctrl+C.")
        while True:
            connection, address = server.accept()
            with connection:
                try:
                    header, payload, request_wire_bytes = recv_message(connection)
                    if header.get("kind") != "frame":
                        raise ValueError("frameメッセージだけを受信できます。")
                    response = process_frame(header, payload, model, args)
                    response_wire_bytes = send_message(connection, response)
                    print(
                        f"frame={response['frame_id']} from={address[0]} "
                        f"transport={header.get('transport')} "
                        f"request_kb={request_wire_bytes / 1024:.2f} "
                        f"response_kb={response_wire_bytes / 1024:.2f} "
                        f"cloud_ms={response['timings']['cloud_processing_ms']:.3f}"
                    )
                except (ConnectionError, ValueError, cv2.error, OSError) as exc:
                    try:
                        send_error(connection, str(exc))
                    except OSError:
                        pass
                    print(f"Request from {address} failed: {exc}")


if __name__ == "__main__":
    main()
