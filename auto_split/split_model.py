import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
from ultralytics.engine.results import Results
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

class SplitYOLOWrapper:
    def __init__(self, model_name='yolov8n.pt'):

        print(f"Loading model: {model_name} ...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo = YOLO(model_name)
        self.model = self.yolo.model.to(self.device)
        # BatchNorm / Dropout の挙動を推論モードにする
        self.model.eval()

        self.layers = list(self.model.model)
        # 各skipの最終使用位置を計算
        # m.fのfは接続情報from(どこのレイヤーから来るか)
        self.last_use = {}
        for i, m in enumerate(self.layers):
            if m.f != -1:
                sources = m.f if isinstance(m.f, list) else [m.f]
                for s in sources:
                    # -1(直前のレイヤー)は除外
                    if s >= 0:
                        self.last_use[s] = i


    # -------------------------
    # 画像前処理
    # -------------------------
    def preprocess(self, img_path):
        # JPEG / PNG → NumPy配列
        img0 = cv2.imread(img_path) # OpenCVは BGR順
        # 画像ではrow = y（縦）, col = x（横）のため、H, Wの順番
        h0, w0 = img0.shape[:2] # (H0, W0, 3)から(H0, W0)を取り出す

        # YOLOと同じletterbox
        # new_shape : 出力サイズ, stride: モデルストライド整合用
        letterbox = LetterBox(new_shape=640, stride=32) # 640*640の正方形: 画像の長辺に合わせて余白は0で埋める

        img = letterbox(image=img0) # image: 入力画像

        # -------------------------
        # meta情報計算
        # -------------------------
        h, w = img.shape[:2]

        scale = min(640 / h0, 640 / w0)
        pad_x = (640 - w0 * scale) / 2
        pad_y = (640 - h0 * scale) / 2

        meta = {
            "scale": scale,
            "pad": (pad_x, pad_y),
            "orig_shape": (h0, w0),
            "input_shape": (h, w)
        }

        img = img[:, :, ::-1]  # BGR → RGB
        img = img.transpose(2, 0, 1) # (H, W, C) → (C, H, W)=Pytorch仕様
        img = np.ascontiguousarray(img) # メモリを連続配置してPyTorch tensor変換高速化

        img = torch.from_numpy(img).float() / 255.0 # 勾配爆発を防ぐため # floatを明示
        # バッチ次元追加: (3, 640, 640)→(1, 3, 640, 640)=YOLOの入力形式(N, C, H, W)
        img = img.unsqueeze(0).to(self.device) # .to(self.device)でデバイス転送

        return img, img0, meta ### img0は必要？

    # -------------------------
    # Edge
    # -------------------------
    def run_edge(self, x, split_index, meta):
        # print(self.model.save) [4, 6, 9, 12, 15, 18, 21]
        y = []
        cnt = 0

        for i, m in enumerate(self.layers):

            if i > split_index:
                break
            # skip/concat
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in) # ニューラルネット1層のforward計算

            # -------------------------
            # 必要なcontextだけ保存
            # -------------------------
            # if i in self.last_use and i <= split_index < self.last_use[i]: # i <=はおそらくいらない
            #    y.append(x)
            # else:
            #     y.append(None)
            if i in self.model.save:
                y.append(x)
                cnt += 1 
            else:
                y.append(None)
            # self.model.saveは「スキップ接続で再利用される層番号」のリスト
            # y.append(x if m.i in self.model.save else None)
            # -> Edgeで計算済みの中間テンソルまで全て保存されてしまう
        # x=Edgeの最終出力テンソル, y=中間保存テンソル群
        # metaも一緒に返す
        print(f"中間特徴量個数{cnt}")
        return x, y, meta

    # -------------------------
    # Cloud
    # -------------------------
    def run_cloud(self, x, saved_y, split_index):

        y = saved_y.copy()

        for i in range(split_index + 1, len(self.layers)):

            m = self.layers[i]

            if m.f != -1:

                if isinstance(m.f, int):
                    x_in = y[m.f]

                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]

            else:
                x_in = x

            x = m(x_in)

            if i in self.model.save:
                y.append(x)
            else:
                y.append(None)
            
            print("layer", i, "f=", m.f)
            print("x shape", x.shape if isinstance(x, torch.Tensor) else None)
            print("inputs", [type(t) for t in x_in] if isinstance(x_in,list) else type(x_in))

        return x