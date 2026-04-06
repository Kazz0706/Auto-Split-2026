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
        self.model.eval()

        self.layers = list(self.model.model)


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

        return img, img0, meta


    # -------------------------
    # Edge
    # -------------------------
    def run_edge(self, x, split_index, meta):
        # print(self.model.save) [4, 6, 9, 12, 15, 18, 21]
        y = []

        for i, m in enumerate(self.layers):

            if i > split_index:
                break

            if m.f != -1:
                if isinstance(m.f, int): # 単一入力の場合
                    x_in = y[m.f]
                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            # self.model.saveは「スキップ接続で再利用される層番号」のリスト
            y.append(x if m.i in self.model.save else None)

        # x=Edgeの最終出力テンソル, y=中間保存テンソル群
        # metaも一緒に返す
        return x, y, meta

    # -------------------------
    # Cloud
    # -------------------------
    def run_cloud(self, x, saved_y, split_index, meta):

        y = saved_y # 層番号のリスト

        for i, m in enumerate(self.layers):

            if i <= split_index:
                continue

            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in) # 重み・演算内容・接続情報を持つNNの1層
            y.append(x if m.i in self.model.save else None)

        return x, meta


# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
if __name__ == "__main__":

    wrapper = SplitYOLOWrapper('yolov8n.pt')

    img_path = "images/test.jpg"

    # 前処理
    # img = wrapper.preprocess(img_path)
    img, orig_img, meta = wrapper.preprocess(img_path)

    split_point = 5

    with torch.no_grad():

        print("Edge...")
        edge_out, context, meta = wrapper.run_edge(img, split_point, meta)

        print("Cloud...")
        final_result, meta = wrapper.run_cloud(edge_out, context, split_point, meta)

    print("\n===== RESULT =====")

    if isinstance(final_result, (list, tuple)):
        for i, t in enumerate(final_result):
            if isinstance(t, torch.Tensor):
                print(f"Tensor {i} shape:", t.shape)
    else:
        print("Output shape:", final_result.shape)

    # -----------------------------
    # YOLO形式に変換して保存
    # -----------------------------

    pred = final_result[0] if isinstance(final_result, (list, tuple)) else final_result

    # NMS: 重複削除(確率が最も高いboxだけ残す)
    # conf_thres: 確率の閾値、iou_thres: boxの重なり度(これを越えたら片方削除)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 元画像
    orig_img = cv2.imread(img_path)

    # h0, w0 = orig_img.shape[:2]   # 元サイズ
    # h, w = 640, 640               # 推論サイズ

    results = []

    for det in pred:
        # if len(det):
        #     # 🔥 ここが超重要
        #     det[:, :4] = scale_boxes(
        #         (h, w),
        #         det[:, :4],
        #         (h0, w0)
        #     )

        # det[:, :4]で[x1, y1, x2, y2, conf, class]から前半4列(全boxの座標部分)取り出す
        # すなわち、N=boxの数として、N*6→N*4に変換
        # scale_boxesでboxesを元画像座標に変換
        det[:, :4] = scale_boxes(
            meta["input_shape"],      # (640, 640)
            det[:, :4],
            meta["orig_shape"]        # (H0, W0)
        ) # 引数は(モデル入力サイズ, 全boxの座標, 元画像サイズ)

        r = Results(
            orig_img=orig_img,
            path=img_path,
            names=wrapper.yolo.names,
            boxes=det
        )

        # meta情報をResultsに追加（研究用）
        r.meta = meta

        results.append(r)

    # 描画して保存
    for r in results:
        plotted = r.plot()   # ← 枠と確率を描画
        cv2.imwrite("split_result.jpg", plotted)

    print("Saved: split_result.jpg")

    # meta情報(pad, scale)を含めてedgeからcloudに送る
    # best_splitの位置によって無駄な中間特徴量転送を省く
    # 例）層1→層3→層5, best_split=4の場合、層1は要らない情報