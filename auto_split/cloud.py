# cloud.py
import pickle
import struct
import socket
import torch
import cv2
from split_model import SplitYOLOWrapper
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.engine.results import Results

wrapper = SplitYOLOWrapper("yolov8n.pt")

# 進捗：Docker HubにDocker Imageをpush&pull → ラズパイ上でMacと同じ実験環境を実現
# ラズパイ、Mac間でsocketを用いたTCP通信を行う → 分割推論にも成功
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
    s.bind((HOST, PORT))
    s.listen(1)
    print("Waiting for connection...")

    conn, addr = s.accept()
    with conn:
        print("Connected:", addr)

        # 4byteで長さ取得
        header = recvall(conn, 4) # headerで指定された長さのバイト列を取得
        msg_len = struct.unpack(">I", header)[0] # タプルのうち1番目（データ長）を取得

        # 本体受信
        data = recvall(conn, msg_len)
# バイト列 -> Pythonのオブジェクト(bytes → dict（元のpacket）)
packet = pickle.loads(data)

# with open("packet.pkl", "rb") as f:
#     packet = pickle.load(f)

edge_out_q = packet["edge_out"]
scale_q = packet["scale_q"]

# -------------------------
# 復元（dequantize）
# -------------------------
edge_out = edge_out_q.float() * scale_q

edge_out = edge_out.to(wrapper.device)

context = packet["context"]
meta = packet["meta"]
split_point = packet["split"]

with torch.inference_mode():
    final_result, meta = wrapper.run_cloud(edge_out, context, split_point, meta)

# 🔥 inference tensor → normal tensor(不足している情報は初期化状態で追加される)
pred = final_result[0] if isinstance(final_result, (list, tuple)) else final_result
pred = pred.detach().clone()

# NMS
pred = non_max_suppression(pred)

orig_img = cv2.imread("images/test.jpg")

for det in pred:

    if det is None or len(det) == 0:
        continue

    det[:, :4] = scale_boxes(
        meta["input_shape"],
        det[:, :4],
        meta["orig_shape"]
    )

    r = Results(
        orig_img=orig_img,
        path="img",
        names=wrapper.yolo.names,
        boxes=det
    )

    plotted = r.plot()
    cv2.imwrite("result.jpg", plotted)