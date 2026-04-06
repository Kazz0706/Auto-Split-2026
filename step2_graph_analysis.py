import torch
from ultralytics import YOLO
import torch.fx

def analyze_yolo_graph():
    # 1. モデルロード
    model = YOLO("yolov8n.pt")
    
    # 2. PyTorchモデル本体を取り出す
    # YOLOv8は .model という属性にnn.Moduleを持っています
    pytorch_model = model.model
    
    # 3. ダミー入力でトレース（グラフ化）する
    # ※ torch.fxだけでは複雑なYOLOを解析しきれない場合があるため、
    #    簡易的に各層の名前と出力を確認するフックを仕掛けます。
    
    print(f"{'Layer Name':<25} | {'Output Shape':<20} | {'Size (KB)':<10}")
    print("-" * 65)

    def get_activation(name):
        def hook(model, input, output):
            # outputがタプルの場合もあるのでケア
            if isinstance(output, tuple):
                o = output[0]
            else:
                o = output
            
            # 形状取得
            shape = list(o.shape)
            # サイズ計算 (要素数 * 4byte / 1024 = KB)
            numel = o.numel()
            size_kb = (numel * 4) / 1024
            
            print(f"{name:<25} | {str(shape):<20} | {size_kb:.1f} KB")
        return hook

    # 全ての層にフック（監視装置）を取り付ける
    hooks = []
    for name, layer in pytorch_model.named_modules():
        # Conv層やBottleneck層など、主要な層だけ見る
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
             h = layer.register_forward_hook(get_activation(name))
             hooks.append(h)

    # 4. ダミーデータを通して計測
    dummy_input = torch.zeros(1, 3, 640, 640)
    pytorch_model(dummy_input)

    # クリーンアップ
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    analyze_yolo_graph()

# 現在、あなたはテキストとしてログを見ていますが、Auto-Splitのプログラムではこれを数値として取得する必要があります。
# torch.fx を使うと、このテキストログのような情報を、プログラム上の「グラフ（ノードのつながり）」として取得できます。
# 次に実行すべきコード（step2_graph_analysis.py）を作成しました。これを実行して、「分割候補」をリスト化してみましょう。
# このコードを実行すると、先ほどのテキストログよりもさらに具体的な**「層の名前」と「データサイズ（KB）」のリスト**が出力されます。これがAuto-Splitの入力データになります。