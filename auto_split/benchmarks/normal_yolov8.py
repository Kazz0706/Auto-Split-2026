from ultralytics import YOLO
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

def main():

    model_path = PROJECT_DIR / "models" / "yolov8n.pt"
    img_path = PROJECT_DIR / "samples" / "test.jpg"
    save_path = PROJECT_DIR / "outputs" / "normal_result.jpg"

    # モデル読み込み
    model = YOLO(model_path)

    # 推論
    results = model.predict(
        source=img_path,
        conf=0.25,
        iou=0.45,
        device="cpu"   # GPUなら0, CPUなら"cpu"
    )

    # 画像描画
    r = results[0]
    plotted = r.plot()

    # 保存
    save_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(save_path), plotted)

    print("Saved:", save_path)


if __name__ == "__main__":
    main()
