"""クラウド側のYOLOv8分割推論サーバー。"""

from __future__ import annotations

import io
import json
import os
import pickle
import socket
import struct
import time
from pathlib import Path

import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from auto_split.src.split_model import SplitYOLOWrapper


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"
HOST = "0.0.0.0"
PORT = int(os.environ.get("CLOUD_PORT", "5001"))


def recvall(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("エッジ側が要求の途中で接続を閉じました。")
        chunks.extend(chunk)
    return bytes(chunks)


def process_packet(packet: dict, wrapper: SplitYOLOWrapper) -> dict:
    restore_start = time.perf_counter()
    meta = json.loads(packet["meta"].decode())
    meta["input_shape"] = list(meta["input_shape"])
    meta["orig_shape"] = list(meta["orig_shape"])
    split_point = int(packet["split"])

    scale_q = max(float(packet["scale_q"]), 1e-8)
    edge_out_q = torch.load(io.BytesIO(packet["tensor"]), map_location="cpu")
    edge_out = (edge_out_q.float() * scale_q).to(wrapper.device)

    context_tensors = torch.load(io.BytesIO(packet["context"]), map_location="cpu")
    context = [None] * (split_point + 1)
    for index, tensor, scale in zip(packet["context_idx"], context_tensors, packet["context_scales"]):
        context[int(index)] = (tensor.float() * max(float(scale), 1e-8)).to(wrapper.device)
    restore_ms = (time.perf_counter() - restore_start) * 1000

    cloud_start = time.perf_counter()
    with torch.inference_mode():
        final_result = wrapper.run_cloud(edge_out, context, split_point)
    pred = final_result[0] if isinstance(final_result, (list, tuple)) else final_result
    pred = non_max_suppression(pred.detach().clone())

    boxes = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        det = det.detach().clone()
        det[:, :4] = scale_boxes(meta["input_shape"], det[:, :4], meta["orig_shape"])
        boxes.extend(det.cpu().numpy().tolist())
    cloud_time = time.perf_counter() - cloud_start
    print(f"restore_ms={restore_ms:.3f} cloud_processing_ms={cloud_time * 1000:.3f}")
    return {"boxes": boxes, "meta": meta, "cloud_time": cloud_time}


def main() -> None:
    wrapper = SplitYOLOWrapper(MODEL_PATH)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"Cloud server listening on {HOST}:{PORT}. Stop with Ctrl+C.")
        while True:
            connection, address = server.accept()
            with connection:
                try:
                    request_size = struct.unpack(">I", recvall(connection, 4))[0]
                    packet = pickle.loads(recvall(connection, request_size))
                    response = pickle.dumps(process_packet(packet, wrapper))
                    connection.sendall(struct.pack(">I", len(response)) + response)
                    print(f"Processed request from {address}.")
                except (ConnectionError, EOFError, pickle.UnpicklingError) as exc:
                    print(f"Request from {address} failed: {exc}")


if __name__ == "__main__":
    main()
