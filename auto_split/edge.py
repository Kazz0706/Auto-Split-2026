# edge.py
import time
import json
import io
import numpy as np
import torch
import pickle
import struct
import socket
import cv2
from ultralytics.engine.results import Results
from split_model import SplitYOLOWrapper   # 共通クラス

wrapper = SplitYOLOWrapper("yolov8n.pt")

img_path = "images/test.jpg"

# -------------------------
# Timer start
# -------------------------
A_start = time.perf_counter()
B_start = A_start

# 前処理
img, orig_img, meta = wrapper.preprocess(img_path)

split_point = 20
print(f"split_point: {split_point}")

# 勾配計算（＝学習用の計算）を完全にオフにする→計算速度向上, メモリ削減(全中間テンソルを保存しないため)
with torch.inference_mode():
    edge_out, context, meta = wrapper.run_edge(img, split_point, meta)

print("save layers:", wrapper.model.save)

print("context positions:")
for i,t in enumerate(context):
    if t is not None:
        print(i, t.shape)

# -------------------------
# 量子化（通信量削減）
# -------------------------
# float32 -> INT8: 通信量75%削減
max_val = edge_out.abs().max()
scale_q = max_val / 127
# INT8 tensorに変更
edge_out_q = torch.round(edge_out / scale_q).to(torch.int8)

# -------------------------
# Tensor → torch.save (高速)
# -------------------------
tensor_buffer = io.BytesIO()
torch.save(edge_out_q.cpu(), tensor_buffer)
tensor_bytes = tensor_buffer.getvalue()

# -------------------------
# context → torch.save
# -------------------------
context_buffer = io.BytesIO()

# Noneを除いて保存 → 通信量削減, Noneのpickle経由防止
context_indices = []
context_tensors = []

for i, t in enumerate(context):
    if t is not None:
        context_indices.append(i)
        context_tensors.append(t.cpu())

# tensor list を保存
torch.save(context_tensors, context_buffer)
context_bytes = context_buffer.getvalue()

meta_bytes = json.dumps(meta).encode() # metaをpickle → json

# 送信パケット
packet = {
    "tensor": tensor_bytes,
    "context": context_bytes,           # bytes
    "context_idx": np.array(context_indices, dtype=np.int32),     # list
    "meta": meta_bytes,
    "scale_q": scale_q.cpu().numpy(), # 復元用
    "split": split_point
}

# -------------------------
# socket送信(ラズパイ→Mac)
# -------------------------
HOST = "192.168.12.29"  # ← MacのIP
PORT = 5001 # ← MacのPORT

data = pickle.dumps(packet)

# 長さヘッダ（4byte）
header = struct.pack(">I", len(data))

A_end = time.perf_counter()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(header + data)  # 🔥 まとめて送る

    # -------------------------
    # Cloud結果受信
    # -------------------------
    def recvall(sock, n):
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    header = recvall(s, 4)
    msg_len = struct.unpack(">I", header)[0]

    data = recvall(s, msg_len)

result_packet = pickle.loads(data)

boxes = result_packet["boxes"]
meta = result_packet["meta"]
cloud_time = result_packet["cloud_time"]

B_end = time.perf_counter()

# Calculate time
D_start = time.perf_counter()

orig_img = cv2.imread(img_path)

# OpenCVで自作描画plot
# for box in boxes:
#     x1,y1,x2,y2,conf,cls = box
#     label = f"{wrapper.yolo.names[int(cls)]} {conf:.2f}"

#     cv2.rectangle(orig_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)

#     cv2.putText(
#         orig_img,
#         label,
#         (int(x1), int(y1)-10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0,255,0),
#         2
#     )

# cv2.imwrite("result.jpg", orig_img)

# Ultralyticsを用いて描画
r = Results(
    orig_img=orig_img,
    path="img_path",
    names=wrapper.yolo.names,
    boxes=torch.tensor(boxes)
)

plotted = r.plot()
cv2.imwrite("result.jpg", plotted)

D_end = time.perf_counter()

# Calculate time
edge_time = A_end - A_start
cloud_time = cloud_time
total_time = B_end - B_start
comm_time = (total_time - edge_time - cloud_time) / 2
plot_time = D_end - D_start

print("Edge time:", edge_time)
print("Cloud time:", cloud_time)
print("Communication time:", comm_time)
print("Edge-Comm-Cloud time:", total_time)
print("Plot time:", plot_time)