from ultralytics import YOLO
import cv2

def main():

    model_path = "yolov8n.pt"          # モデル
    img_path = "images/test.jpg"     # 入力画像
    save_path = "normal_result.jpg"    # 保存先

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
    cv2.imwrite(save_path, plotted)

    print("Saved:", save_path)


if __name__ == "__main__":
    main()