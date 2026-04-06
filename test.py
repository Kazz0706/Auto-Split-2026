# 必要なライブラリ: pip install torch torchinfo ultralytics
import torch
from ultralytics import YOLO
from torchinfo import summary

# 1. 最新のYOLOモデルをロード (重みも自動ダウンロードされます)
model = YOLO('yolov8n.pt') 

# 2. モデルの構造と、各層のパラメータ数、出力サイズを一瞬で表示
# (input_sizeは [Batch, Channel, Height, Width])
# summary(model.model, input_size=(1, 3, 640, 640))

# from ultralytics import YOLO

# # 1. YOLOv8 nanoモデルをロード
# model = YOLO("yolov8n.pt")

# 2. ONNX形式にエクスポート（構造を可視化するためのファイル形式）
# opset=12あたりが互換性が良くて無難です
path = model.export(format="onnx", opset=12)

print(f"モデルの図面ファイルを作成しました: {path}")
# https://netron.app/から図を確認できる