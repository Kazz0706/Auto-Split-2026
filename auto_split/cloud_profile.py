import time
import torch
from split_model import SplitYOLOWrapper

wrapper = SplitYOLOWrapper("yolov8n.pt")

img_path = "images/test.jpg"

img, orig_img, meta = wrapper.preprocess(img_path)

split_layer = -1

x = img
y = []
total = 0

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
        t0 = time.perf_counter()

        x = m(x_in)

        t1 = time.perf_counter()

        layer_time = (t1 - t0) * 1000

        total += layer_time

        print(f"Layer {i:2d} | {type(m).__name__:20s} | {layer_time:.3f} ms")

        # save skip tensor
        if i in wrapper.model.save:
            y.append(x)
        else:
            y.append(None)
    
    print(f"Total time{total}")