import time
import torch
from split_model import SplitYOLOWrapper

load_start = time.perf_counter()
wrapper = SplitYOLOWrapper("yolov8n.pt")
load_end = time.perf_counter()

img_path = "images/test.jpg"

img, orig_img, meta = wrapper.preprocess(img_path)

preprocess_end = time.perf_counter()

split_layer = -1

x = img
y = []
total_inference_time = 0

print("\n===== CLOUD PROFILE =====")

with torch.inference_mode():

    for i, m in enumerate(wrapper.layers):

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
        layer_start = time.perf_counter()

        x = m(x_in)

        layer_end = time.perf_counter()

        layer_time = (layer_end - layer_start) * 1000

        total_inference_time += layer_time

        print(f"Layer {i:2d} | {type(m).__name__:20s} | {layer_time:.3f} ms")

        # save skip tensor
        if i in wrapper.model.save:
            y.append(x)
        else:
            y.append(None)
    
    end_time = time.perf_counter()

    print(f"Model loading time: {(load_end - load_start)*1000:.3f} ms")
    print(f"Preprocessing time: {(preprocess_end - load_end)*1000:.3f} ms")
    print(f"Total inference time: {total_inference_time:.3f} ms")
    print(f"Total cloud time(except model loading): {(end_time - load_end)*1000:.3f} ms")