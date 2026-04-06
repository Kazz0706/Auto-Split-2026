import torch
import time
import pickle
import numpy as np
from split_model import SplitYOLOWrapper


# -------------------------
# 設定
# -------------------------

MODEL = "yolov8n.pt"
IMG = "images/test.jpg"

# 帯域 (bits/sec)
BANDWIDTH = 50 * 1024 * 1024   # 50Mbps

wrapper = SplitYOLOWrapper(MODEL)

img, orig_img, meta = wrapper.preprocess(IMG)

num_layers = len(wrapper.layers)

results = []

print("\n===== Auto-Split Search =====\n")

# 0-9層：backbone → Tensor型, 10-17層：neck → Tensor型, 18-22層：detect head → tuple型
for split in range(num_layers-3):

    print(f"\n--- Split {split} ---")

    with torch.inference_mode():

        # -----------------
        # EDGE TIME
        # -----------------
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.time()

        edge_out, context, meta_edge = wrapper.run_edge(img, split, meta)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        edge_time = time.time() - t0


        # -----------------
        # 通信量
        # -----------------

        packet = {
            "edge_out": edge_out.cpu(),
            "context": context,
            "meta": meta_edge,
            "split": split
        }

        data = pickle.dumps(packet)
        data_size = len(data)  # bytes

        trans_time = (data_size * 8) / BANDWIDTH


        # -----------------
        # CLOUD TIME
        # -----------------

        edge_out_cloud = edge_out.to(wrapper.device)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t1 = time.time()

        final_result, meta_cloud = wrapper.run_cloud(
            edge_out_cloud,
            context,
            split,
            meta_edge
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        cloud_time = time.time() - t1


        total = edge_time + trans_time + cloud_time

        print("edge_time :", edge_time)
        print("cloud_time:", cloud_time)
        print("comm_time :", trans_time)
        print("feature size (KB):", data_size / 1024)
        print("total:", total)


        results.append({
            "split": split,
            "edge_time": edge_time,
            "cloud_time": cloud_time,
            "comm_time": trans_time,
            "size": data_size,
            "total": total
        })


# -------------------------
# BEST SPLIT
# -------------------------

best = min(results, key=lambda x: x["total"])

print("\n==============================")
print("BEST SPLIT:", best["split"])
print("TOTAL LATENCY:", best["total"])
print("==============================")

# 表形式表示
print("\nAll Results:\n")

for r in results:
    print(
        f"split={r['split']:2d} | "
        f"edge={r['edge_time']:.4f}s | "
        f"cloud={r['cloud_time']:.4f}s | "
        f"comm={r['comm_time']:.4f}s | "
        f"size={r['size']/1024:.1f}KB | "
        f"total={r['total']:.4f}s"
    )