import torch
import cv2
import time
import struct
import numpy as np
from ultralytics import YOLO
# from ultralytics.utils.ops import non_max_suppression
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
        
    def get_needed_context(self, split_index):

        needed = set()

        for i in range(split_index + 1, len(self.layers)):

            m = self.layers[i]

            if m.f != -1:

                src = m.f if isinstance(m.f, list) else [m.f]

                for s in src:

                    if s >= 0 and s < split_index:
                        needed.add(s)

        return needed


    # -------------------------
    # 画像前処理
    # -------------------------
    def preprocess(self, img_path):
        """画像ファイルを読み込み、共通のフレーム前処理へ渡す。"""
        read_start = time.perf_counter()
        img0 = cv2.imread(str(img_path))
        read_end = time.perf_counter()
        if img0 is None:
            raise FileNotFoundError(f"画像を読み込めません: {img_path}")

        img, orig_img, meta = self.preprocess_frame(img0)
        self.last_preprocess_timing["image_read_ms"] = (read_end - read_start) * 1000
        print(f"image read_ms: {self.last_preprocess_timing['image_read_ms']:.3f}")
        return img, orig_img, meta

    def preprocess_frame(self, img0):
        """BGRのNumPyフレームをYOLOv8入力テンソルへ変換する。

        Picamera2やOpenCVカメラのフレームをファイル保存せず、そのまま
        分割推論の前処理へ渡すための共通経路。
        """
        if img0 is None or img0.ndim != 3 or img0.shape[2] != 3:
            raise ValueError("img0 は形状 (H, W, 3) のBGR NumPy配列である必要があります。")

        start = time.perf_counter()
        print(f"Original image shape: {img0.shape}")
        h0, w0 = img0.shape[:2]

        letterbox = LetterBox(new_shape=640, stride=32)
        img = letterbox(image=img0)
        letterbox_end = time.perf_counter()

        h, w = img.shape[:2]
        scale = min(640 / h0, 640 / w0)
        pad_x = (640 - w0 * scale) / 2
        pad_y = (640 - h0 * scale) / 2
        meta = {
            "scale": scale,
            "pad": (pad_x, pad_y),
            "orig_shape": (h0, w0),
            "input_shape": (h, w),
        }

        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        layout_end = time.perf_counter()

        img = torch.from_numpy(img).float() / 255.0
        tensor_end = time.perf_counter()

        img = img.unsqueeze(0).to(self.device)
        device_end = time.perf_counter()

        self.last_preprocess_timing = {
            "letterbox_ms": (letterbox_end - start) * 1000,
            "layout_conversion_ms": (layout_end - letterbox_end) * 1000,
            "tensor_conversion_ms": (tensor_end - layout_end) * 1000,
            "device_transfer_ms": (device_end - tensor_end) * 1000,
            "total_ms": (device_end - start) * 1000,
        }
        for name, value in self.last_preprocess_timing.items():
            print(f"preprocess {name}: {value:.3f} ms")

        return img, img0, meta

    # -------------------------
    # Edge
    # -------------------------
    def run_edge(self, x, split_index):
        y = [None] * (split_index + 1)  # レイヤー番号でアクセスできるリスト
        x_history = [None] * (split_index + 1)
        needed = self.get_needed_context(split_index)

        for i, m in enumerate(self.layers):
            if i > split_index:
                break
            # skip/concat
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = x_history[m.f]
                    if m.f in self.last_use and self.last_use[m.f] == i:
                        x_history[m.f] = None # 最後の使用であればメモリ解放
                else:
                    x_in = []
                    for j in m.f:
                        if j == -1:
                            x_in.append(x)
                        else:
                            x_in.append(x_history[j])
                            if self.last_use[j] == i:
                                x_history[j] = None # 最後の使用であればメモリ解放
            else:
                x_in = x

            x = m(x_in) # ニューラルネット1層のforward計算

            # -------------------------
            # 必要なcontextだけ保存
            # -------------------------
            if i in needed:
                y[i] = x

            if i in self.last_use and i < self.last_use[i] <= split_index:
                x_history[i] = x

        # x=Edgeの最終出力テンソル, y=中間保存テンソル群
        # metaも一緒に返す
        print(f"中間特徴量{needed}")
        return x, y

    # -------------------------
    # Cloud
    # -------------------------
    def run_cloud(self, x, saved_y, split_index):

        y = saved_y.copy()
        if split_index in self.model.save:
            y[split_index] = x

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
            
            # print("layer", i, "f=", m.f)
            # print("x shape", x.shape if isinstance(x, torch.Tensor) else None)
            # print("inputs", [type(t) for t in x_in] if isinstance(x_in,list) else type(x_in))

        return x
