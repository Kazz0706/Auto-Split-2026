import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# クラス定義: 分割推論ラッパー
# ---------------------------------------------------------
class SplitYOLOWrapper:
    def __init__(self, model_name='yolov8n.pt'):
        print(f"Loading model: {model_name} ...")
        self.yolo = YOLO(model_name)
        self.model = self.yolo.model
        self.layers = list(self.model.model) # レイヤーのリスト
        
    # --- [機能1] 最適な分割点を探す (プロファイリング) ---
    def find_best_split_point(self, dummy_input):
        print("\n=== Profiling: Searching for the best split point ===")
        
        # 1. 各レイヤーの出力サイズを計測
        layer_sizes = {}
        def get_size_hook(idx):
            def hook(module, input, output):
                # 出力がリストやタプルの場合も考慮してサイズ計算
                if isinstance(output, (list, tuple)):
                    size = sum(t.numel() * t.element_size() for t in output if isinstance(t, torch.Tensor))
                elif isinstance(output, torch.Tensor):
                    size = output.numel() * output.element_size()
                else:
                    size = 0
                layer_sizes[idx] = size
            return hook

        hooks = []
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(get_size_hook(i)))
        
        # ダミー実行（サイズ計測）
        with torch.no_grad():
            self.model(dummy_input)
        
        for h in hooks: h.remove() # フック解除

        # 2. 通信量を計算
        # YOLOv8の各レイヤーは「m.f (from)」という属性で、どのレイヤーの入力を使うか知っている
        total_layers = len(self.layers)
        comm_costs = {}

        print(f"{'Split Layer':<12} | {'Transmission Size (MB)':<25}")
        print("-" * 45)

        # 最後のDetect層(total_layers-1)の手前まで走査
        for k in range(total_layers - 1):
            transmission_bytes = 0
            
            # 「切断点 k」より未来(k+1以降)にあるレイヤーが、
            # 「切断点 k」以前(0〜k)のレイヤーの出力を必要としているかチェック
            for future_idx in range(k + 1, total_layers):
                m = self.layers[future_idx]
                
                # m.f は「入力元のインデックス」 (int または list[int])
                if m.f == -1: 
                    # 直前のレイヤーを見る場合 -> future_idx - 1
                    sources = [future_idx - 1]
                elif isinstance(m.f, int):
                    sources = [m.f]
                elif isinstance(m.f, list):
                    sources = m.f
                else:
                    sources = []

                # ソースの中に「k以下（過去）」のものが含まれていれば、そのデータは送信が必要
                for src in sources:
                    # 相対指定(-1など)の解決は ultralytics 内部で行われているが
                    # ここでは m.f が絶対インデックスになっている前提で計算
                    # (YOLOv8の m.f は通常絶対インデックスに変換済み)
                    if src != -1 and src <= k:
                         if src in layer_sizes:
                             transmission_bytes += layer_sizes[src]
                    elif src == -1 and (future_idx - 1) <= k:
                         # 直前指定で、かつその直前がk以下の場合
                         prev = future_idx - 1
                         if prev in layer_sizes:
                             transmission_bytes += layer_sizes[prev]

            # 重複してカウントしないようにセットなどで管理すべきだが、
            # 簡易的に「必要なレイヤーの出力サイズの総和」とする
            # (厳密には同じテンソルを複数回送る必要はないので、テンソルIDで管理するのがベストだが、近似値としてこれでOK)
            
            mb_size = transmission_bytes / 1024 / 1024
            comm_costs[k] = mb_size
            
            # 候補として表示（全部出すと長いので適当に間引くか、重要なとこだけ見る）
            if k % 1 == 0: 
                print(f"Layer {k:<6} | {mb_size:.4f} MB")

        # 最小コストのレイヤーを探す
        best_layer = min(comm_costs, key=comm_costs.get)
        print("-" * 45)
        print(f"Best Split Layer found: {best_layer} (Size: {comm_costs[best_layer]:.4f} MB)")
        return best_layer

    # --- [機能2] エッジ側実行 ---
    def run_edge(self, x, split_index):
        y = []
        for i, m in enumerate(self.layers):
            if i > split_index: break
            
            # 入力準備
            if m.f != -1:
                if isinstance(m.f, int): x_in = y[m.f]
                else: x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x
            
            x = m(x_in)
            y.append(x if m.i in self.model.save else None)
        
        return x, y

    # --- [機能3] クラウド側実行 ---
    def run_cloud(self, x, saved_y, split_index):
        y = saved_y
        for i, m in enumerate(self.layers):
            if i <= split_index: continue
            
            if m.f != -1:
                if isinstance(m.f, int): x_in = y[m.f]
                else: x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x
            
            x = m(x_in)
            y.append(x if m.i in self.model.save else None)
            
        return x

# ---------------------------------------------------------
# メイン実行部
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. ラッパーの初期化
    wrapper = SplitYOLOWrapper('yolov8n.pt')
    
    # ダミー画像
    img = torch.randn(1, 3, 640, 640)

    # 2. 自動で最適な分割点を探す
    # (これがあなたのCode1の役割です)
    best_split_point = wrapper.find_best_split_point(img)

    # 3. 見つけた分割点で分割推論を実行
    # (これがあなたのCode2の役割です)
    print(f"\n=== Running Split Inference at Layer {best_split_point} ===")
    
    # [Edge]
    print("Edge computing...")
    edge_out, context = wrapper.run_edge(img, best_split_point)
    
    # [Simulated Communication]
    # ここで edge_out と context を pickle 等で送る想定
    print(f"Transmitting data... Context list length: {len(context)}")
    
    # [Cloud]
    print("Cloud computing...")
    final_result = wrapper.run_cloud(edge_out, context, best_split_point)
    
    # print(f"Final Result Shape: {final_result.shape}")
    print("Success!")