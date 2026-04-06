# edge.py
import torch
import pickle
import struct
import socket
from split_model import SplitYOLOWrapper   # 共通クラス

wrapper = SplitYOLOWrapper("yolov8n.pt")

img_path = "images/test.jpg"

img, orig_img, meta = wrapper.preprocess(img_path)

split_point = 4

# 勾配計算（＝学習用の計算）を完全にオフにする→計算速度向上, メモリ削減(全中間テンソルを保存しないため)
with torch.inference_mode():
    edge_out, context, meta = wrapper.run_edge(img, split_point, meta)

# -------------------------
# 量子化（通信量削減）
# -------------------------
# float32 -> INT8: 通信量75%削減
max_val = edge_out.abs().max()
scale_q = max_val / 127

edge_out_q = torch.round(edge_out / scale_q).to(torch.int8)

# 送信パケット
packet = {
    "edge_out": edge_out_q.cpu(),   # INT8 tensor
    "scale_q": scale_q.cpu(),       # 復元用
    "context": context,
    "meta": meta,
    "split": split_point
}

# 例：ファイル送信（実際はsocketやHTTP）
# 完成後削除(I/O遅い)
# with open("packet.pkl", "wb") as f:
#     pickle.dump(packet, f)

# -------------------------
# socket送信(ラズパイ→Mac)
# -------------------------
HOST = "192.168.12.30"  # ← MacのIP
PORT = 5001 # ← MacのPORT

data = pickle.dumps(packet)

# 長さヘッダ（4byte）
header = struct.pack(">I", len(data))

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(header + data)  # 🔥 まとめて送る