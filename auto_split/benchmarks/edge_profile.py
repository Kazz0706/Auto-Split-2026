import time
import torch
from pathlib import Path
from auto_split.src.split_model import SplitYOLOWrapper

PROJECT_DIR = Path(__file__).resolve().parents[1]

t1 = time.perf_counter()
wrapper = SplitYOLOWrapper(PROJECT_DIR / "models" / "yolov8n.pt")
t2 = time.perf_counter()

img_path = PROJECT_DIR / "samples" / "test.jpg"

img, orig_img, meta = wrapper.preprocess(img_path)

split_layer = 22   # YOLOv8n最終層

x = img
y = []
total_inference_time = 0

t3 = time.perf_counter()

print("\n===== EDGE PROFILE =====")

with torch.inference_mode():

    for i, m in enumerate(wrapper.layers):

        if i > split_layer:
            break

        # -------------------------
        # skip / concat
        # -------------------------
        if m.f != -1:

            if isinstance(m.f, int):
                x_in = y[m.f]

            else:
                x_in = [x if j == -1 else y[j] for j in m.f]

        else:
            x_in = x

        # -------------------------
        # layer time
        # -------------------------
        tl0 = time.perf_counter()

        x = m(x_in)

        tl1 = time.perf_counter()

        layer_time = (tl1 - tl0) * 1000

        total_inference_time += layer_time

        print(f"Layer {i:2d} | {type(m).__name__:20s} | {layer_time:.3f} ms")

        # save skip tensor
        if i in wrapper.model.save:
            y.append(x)
        else:
            y.append(None)
    t4 = time.perf_counter()

    print(f"Model loading time: {(t2 - t1)*1000:.3f} ms")
    print(f"Preprocessing time: {(t3 - t2)*1000:.3f} ms")
    print(f"total inference time: {total_inference_time}")
    print(f"Total edge time(except model loading): {(t4 - t2)*1000:.3f} ms")
