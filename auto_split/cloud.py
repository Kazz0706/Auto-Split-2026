# cloud.py
import pickle
import struct
import socket
import torch
import cv2
import time
import json
import io
import numpy as np
from split_model import SplitYOLOWrapper
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.engine.results import Results

wrapper = SplitYOLOWrapper("yolov8n.pt")

# 進捗： 計算時間, 通信時間計測する機能を追加
# メタ情報をpickle, 推論テンソルをtorch.saveでptファイルにして送信して、圧縮高速化
# エッジ側で実画像にboxをplotする仕様に変更 → プライバシーの保護
# 中間特徴量をtorch.save()にして送ろうとしたところバグ発生 → バグを修正中

# エッジオンリーの時より速くなっているか？
# それぞれの工夫→定量化
# 関連研究の調査
# Edge&Cloud協調が向いてる分野, 向いてない分野は？

# 通信は「Python(PyTorch)の値をバイナリに変換して送信」→「cloudでバイナリからPythonの値に戻す」
# -------------------------
# socket受信
# -------------------------
HOST = "0.0.0.0"
PORT = 5001

def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print("Waiting for connection...")

    conn, addr = s.accept()
    with conn:
        print("Connected:", addr)

        conn_start = time.perf_counter()

        # 4byteで長さ取得
        header = recvall(conn, 4) # headerで指定された長さのバイト列を取得
        msg_len = struct.unpack(">I", header)[0] # タプルのうち1番目（データ長）を取得

        # 本体受信
        data = recvall(conn, msg_len)
        receive_end = time.perf_counter()

        # バイト列 -> Pythonのオブジェクト(bytes → dict（元のpacket）)
        packet = pickle.loads(data)
        print(f"Receive time: {(receive_end - conn_start)*1000:.3f} ms")

        restore_start = time.perf_counter()

        # -------------------------
        # 復元（dequantize）
        # -------------------------
        tensor_bytes = packet["tensor"]
        meta = json.loads(packet["meta"].decode()) # 研究デバッグ用(Macで可視化する為, 本来は不要)
        meta["input_shape"] = list(meta["input_shape"])
        meta["orig_shape"] = list(meta["orig_shape"])

        # protect against tiny or zero scales from the edge
        eps = 1e-8
        orig_scale_q = float(packet["scale_q"])
        if orig_scale_q <= eps:
            print(f"Warning: received scale_q={orig_scale_q} <= {eps}, clamping to {eps}")
        scale_q = max(orig_scale_q, eps)
        split_point = packet["split"]

        buffer = io.BytesIO(tensor_bytes)
        edge_out_q = torch.load(buffer)

        edge_out = edge_out_q.float() * scale_q
        edge_out = edge_out.to(wrapper.device)

        # context tensor復元
        context_buffer = io.BytesIO(packet["context"])
        context_tensors = torch.load(context_buffer, map_location=wrapper.device)
        context_scales = packet["context_scales"]

        context_idx = packet["context_idx"]

        print("context_idx:", context_idx)

        # 元のcontext構造に復元 (スケール値をイプシロンでクランプして安全化)
        context = [None] * (split_point+1)

        for idx, tensor, scale in zip(context_idx, context_tensors, context_scales):
            s = float(scale)
            if s <= eps:
                print(f"Warning: context scale for idx {int(idx)} is {s} <= {eps}, clamping to {eps}")
                s = eps
            context[int(idx)] = tensor.float() * s
            
        restore_end = time.perf_counter()
        print(f"Restore time: {(restore_end - restore_start)*1000:.3f} ms")

        # print("context length:", len(context))
        # print([type(x) for x in context])

        # Start inference
        C_start = time.perf_counter()
        with torch.inference_mode():
            final_result = wrapper.run_cloud(edge_out, context, split_point)

        # inference tensor → normal tensor(不足している情報は初期化状態で追加される)
        pred = final_result[0] if isinstance(final_result, (list, tuple)) else final_result
        pred = pred.detach().clone()

        nms_start = time.perf_counter()
        # NMS
        pred = non_max_suppression(pred)
        nms_end = time.perf_counter()
        print(f"NMS time: {(nms_end - nms_start)*1000:.3f} ms")

        restore_boxes_start = time.perf_counter()
        boxes = []

        for det in pred:

            if det is None or len(det) == 0:
                continue

            det = det.detach().clone()

            det[:, :4] = scale_boxes(
                meta["input_shape"],
                det[:, :4],
                meta["orig_shape"]
            )

            boxes.extend(det.cpu().numpy().tolist())
        restore_boxes_end = time.perf_counter()
        print(f"Box restore time: {(restore_boxes_end - restore_boxes_start)*1000:.3f} ms")

        C_end = time.perf_counter()
        cloud_time = C_end - C_start

        result_packet = {
            "boxes": boxes,
            "meta": meta,
            "cloud_time": cloud_time
        }

        data = pickle.dumps(result_packet)
        header = struct.pack(">I", len(data))
        # print(f"header: {header}, header_type: {type(header)})")

        send_start = time.perf_counter()
        conn.sendall(header + data)
        send_end = time.perf_counter()
        print(f"Send time: {(send_end - send_start)*1000:.3f} ms")


# Macで元画像にboxをPlotしたい場合
# orig_img = cv2.imread("images/test.jpg")

# for det in pred:

#     if det is None or len(det) == 0:
#         continue

#     det[:, :4] = scale_boxes(
#         meta["input_shape"],
#         det[:, :4],
#         meta["orig_shape"]
#     )

#     r = Results(
#         orig_img=orig_img,
#         path="img",
#         names=wrapper.yolo.names,
#         boxes=det
#     )

#     plotted = r.plot()
#     cv2.imwrite("result.jpg", plotted)