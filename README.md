# Edge-Cloud Collaborative Split Inference Optimization for YOLOv8
(English version is available below.)
## プロジェクト概要 (Project Overview)
本プロジェクトは、エッジデバイス（Raspberry Pi等）とクラウド間（またはクラウドサーバー）でYOLOv8の推論処理を分割実行し、システム全体の遅延と通信コストを最適化する「分割推論フレームワーク」の構築を目的とする。
リソース制約の厳しいエッジ環境とクラウドの計算資源を最適に協調させる分散システムの実現を目指す。
単にモデルを分けるだけでなく、**Skip Connection（層を飛び越えた接続）を考慮したメモリ管理**と**数値安定性を備えた量子化**を統合している点が技術的特徴である。

## 背景と課題 (Background & Challenges)
IoTデバイスの普及に伴い、エッジ側での高度なAI推論が求められているが、単一のエッジデバイスでは計算資源が不足し、クラウドへの全データ転送は通信遅延や帯域幅の圧迫を招く。
本研究では、ニューラルネットワークの推論処理を特定の層で分割し、前半(分割レイヤーまで)をエッジ、後半をクラウドで実行する「分割推論（Split Inference）」を採用する。
最大の課題は、ネットワーク状態や各ノードの計算能力に依存して変動する「通信時間」と「計算時間」の合計を最小化する**最適な分割点（Split Point）の特定**である。

## 手法とアプローチ (Methodology)

本フレームワークでは、以下の2つの柱に基づきエッジ・クラウド協調推論を最適化しています。

1. **実測ベースのプロファイリングと最適化**
    * **実機プロファイリング**: シミュレーションに頼らず、実機間での計算時間・通信時間を分割点ごとに実測し、全体のレイテンシが最小となる分割点を決定。
    * **Docker環境を用いた分割推論**: 分割点決定後は、エッジ、クラウドともに同じDocker環境を用いて分割推論を実行。
2. **計算・通信効率の最大化**
    * **推論モード (`torch.inference_mode()`) の利用**: 勾配計算を無効化し、推論時の計算負荷とメモリ消費を削減する。
    * **中間特徴量の選別・保持**: YOLOv8のSkip Connectionを考慮し、クラウド側で必要となるエッジ側の中間特徴量のみを保持・転送する。
    * **INT8量子化の実装**: 出力テンソルおよび中間特徴量をINT8へ圧縮・復元することで、通信データ量を大幅に削減している。

## 現在の進捗と計測結果 (Current Status & Results)

### 1. プロトタイプ環境
*   **エッジ**: Raspberry Pi 5 (16GB SDRAM, Docker環境)
*   **クラウド**: MacBook Pro (8GB SDRAM, Docker環境)
*   **モデル**: YOLOv8n

### 2. 単体推論時間 (Baseline)
*   **Raspberry Pi 単体**: 357 ms
*   **MacBook 単体**: 131 ms

### 3. 分割推論の実行結果 (Experimental Results)
yolov8n（Ultralytics内部モジュール 22層）において、**Split Layer = 3** を選択した際の計測データを以下に示す。

| 項目 | 計測値 | 備考 |
| :--- | :--- | :--- |
| **トータル処理時間** | **762.8 ms** | 前処理〜推論〜返送まで |
| **前処理 (Preprocessing)** | **284.6 ms** | リサイズ等。**最大のボトルネック** |
| **エッジ側計算時間 (Edge)** | 176.7 ms | レイヤー 0～3 |
| **圧縮処理 (Compression)** | 4.8 ms | INT8量子化・シリアライズ。非常に軽量 |
| **通信 (Communication)** | 46.4 ms | TCP接続確立および双方向送受信時間からの片道近似 |
| **クラウド側計算時間 (Cloud)** | 203.9 ms | レイヤー 4～21 |
| **可視化処理 (Visualization)** | 567.9 ms | 描画処理。トータル処理時間には含めない |
| **通信データサイズ** | 402.44 KB | 片道換算で約8.47 MB/s（推定実効帯域 ≒ 68 Mbps） |

### 4. 分析と考察
*   **ボトルネックの特定**: 前処理（約280ms）が全体の約37%を占めており、ここが最大の最適化ポイントであることが判明した。
*   **分割点の傾向**: Split Layer = 3〜4 付近が現在の環境では最も高速だが、通信・Dockerのオーバーヘッドにより、現状はラズパイ単体での実行（357ms）を上回る遅延が発生している。
*   **量子化の効果**: 圧縮処理自体は5ms以下と非常に高速であり、量子化による精度維持と通信量削減の両立が、実用化に向けた鍵となる。

**[分割推論の出力結果]**
![Output by Split Inference](images/result_v1.jpg)

## 従来研究との差異 (Difference from Prior Work)

分割推論の代表的研究として、Huaweiの **Auto-Split** が挙げられる。  
Auto-Splitは、ResNetやYOLOv3を対象に、エッジ計算・通信・クラウド計算を事前プロファイリングし、総遅延が最小となる分割点（Split Point）を選択する手法である。

しかし、従来研究で主に扱われるResNet系やYOLOv3は、比較的**直線的（Sequential）なネットワーク構造**を前提としている。

一方、YOLOv8は：

* **Skip Connection**
* **FPN / PAN による特徴融合**
* **Concatを伴う有向グラフ（DAG）構造**

を持つ。  
FPN/PANは、画像の細部情報と物体全体の意味情報を複数層で融合し、小物体から大物体まで高精度に検出する仕組みである。その結果、YOLOv8では過去の特徴量が後続層(クラウド側)で再利用されるため、単純に「分割層の出力だけ」を送信しても推論を継続できない。

本研究では、この問題に対し：

* クラウド側で再利用される特徴量のレイヤー番号を事前解析した集合(`needed`)
* テンソル最終使用位置 (`last_use`)
* 必要特徴量のみを保持する `context` 管理

を実装し、**YOLOv8のDAG構造に対応した分割推論**を実現している。

さらに、従来研究が理論帯域（Bandwidth）による通信推定を採用することが多いのに対し、本研究では：

* 実機ソケット通信
* INT8量子化
* Serialization / Deserialization
* Tensor復元

まで含めた**実測ベースの通信時間評価**を行っている。

したがって本研究は、従来の「直線型ネットワーク向け分割推論」を発展させ、**YOLOv8の複雑な特徴融合構造と実通信オーバーヘッドを考慮した、実運用志向のEdge–Cloud分割推論フレームワーク**として位置づけられる。

## YOLOv8nのモデル構造 (YOLOv8n Model Architecture)

YOLOv8n は大きく以下の3つの構成要素から成る。

* **Backbone** – 入力画像から階層的な特徴量を抽出する部分
* **Neck** – FPN/PAN により複数スケールの特徴量を融合し、小物体・大物体双方の検出性能を向上させる部分
* **Head** – 複数スケールの特徴量を用いて最終的な物体検出を行う部分

YOLOv8 は ResNet のような単純な直線型（Sequential）CNNではなく、**Skip Connection** と **Concat による特徴融合** を含む **DAG（Directed Acyclic Graph: 有向非巡回グラフ）構造** を採用している。

この構造は検出精度向上に有効である一方、**分割推論（Split Inference）** においては「過去の特徴量が後段で再利用される」という課題を生む。

### YOLOv8n レイヤー構造 (Ultralytics内部モジュール)

```mermaid
flowchart TD

%% ===== Backbone =====
subgraph Backbone["Backbone（特徴抽出）"]
    L0["0 Conv<br/>3→16"] --> L1["1 Conv<br/>16→32"]
    L1 --> L2["2 C2f<br/>32"]
    L2 --> L3["3 Conv<br/>32→64"]
    L3 --> L4["4 C2f<br/>64"]
    L4 --> L5["5 Conv<br/>64→128"]
    L5 --> L6["6 C2f<br/>128"]
    L6 --> L7["7 Conv<br/>128→256"]
    L7 --> L8["8 C2f<br/>256"]
    L8 --> L9["9 SPPF<br/>256"]
end

%% ===== Neck =====
subgraph Neck["Neck（FPN / PAN特徴融合）"]
    L9 --> L10["10 Upsample"]
    L10 --> L11["11 Concat"]
    L6 --> L11

    L11 --> L12["12 C2f"]

    L12 --> L13["13 Upsample"]
    L13 --> L14["14 Concat"]
    L4 --> L14

    L14 --> L15["15 C2f"]

    L15 --> L16["16 Conv"]
    L16 --> L17["17 Concat"]
    L12 --> L17

    L17 --> L18["18 C2f"]

    L18 --> L19["19 Conv"]
    L19 --> L20["20 Concat"]
    L9 --> L20

    L20 --> L21["21 C2f"]
end

%% ===== Head =====
subgraph Head["Head（検出ヘッド）"]
    L15 --> D["22 Detect"]
    L18 --> D
    L21 --> D
end

%% ===== Colors =====
style Backbone fill:#D6EAF8,stroke:#1F618D,stroke-width:2px
style Neck fill:#D5F5E3,stroke:#1E8449,stroke-width:2px
style Head fill:#FADBD8,stroke:#922B21,stroke-width:2px
```
※ モジュール内の数字は **レイヤー番号** と **チャネル数（Channels）** を表す。  
`A→B` は **入力チャネル数 A、出力チャネル数 B** を意味する。

### 構造の概要

| 構成要素 | 主なモジュール | 役割 |
|---|---|---|
| **Backbone** | Conv, C2f, SPPF | 低レベルから高レベルまでの画像特徴量を抽出 |
| **Neck** | Upsample, Concat, C2f | FPN/PAN によるマルチスケール特徴融合 |
| **Head** | Detect | Bounding Box とクラスの最終予測 |

### 主要モジュールの説明 (Module Descriptions)

| モジュール | 説明 |
|---|---|
| **Conv** | 畳み込みによる特徴抽出とチャネル変換を行う基本層。 |
| **C2f** | Skip Connection を含む YOLOv8 の特徴抽出ブロック。 |
| **SPPF** | 広域コンテキストを効率的に取得するプーリング層。 |
| **Upsample** | 特徴マップの解像度を拡大する処理。 |
| **Concat** | 複数特徴量をチャネル方向に結合する演算。 |
| **Detect** | Bounding Box とクラスを出力する検出ヘッド。 |

### YOLOv8nにおける分割推論の難しさ

YOLOv8 は **完全な直線型モデルではない**。Skip Connection と Concat により、Backbone や Neck の途中で生成された特徴量（例: Layer 4, 6, 9）が後段の層で再利用される。
そのため、分割推論では **「分割層の出力だけ」を送れば良いわけではない**。

例えば：
* **Layer11 ← Layer6**
* **Layer14 ← Layer4**
* **Layer17 ← Layer12**
* **Layer20 ← Layer9**
のような依存関係が存在する。

したがってクラウド側で推論を正しく再開するには、分割層出力 (`edge_out`) に加え、依存解析によって特定された中間特徴量 (`context`) を適切に保持・転送する必要がある。
本研究で実装した `needed`・`last_use`・`context` 管理は、この **YOLOv8のDAG構造に対応したSplit Inference** を実現するための中核技術である(詳細は後述)。

---

## システムアーキテクチャ (System Architecture & Flow)

```mermaid
sequenceDiagram
    participant E as エッジ (Raspberry Pi)
    participant C as クラウド (Mac / GPUサーバー)

    Note over E: 1. 画像読み込み & 前処理 (LetterBox)
    
    rect rgb(240, 248, 255)
    Note over E: 2. エッジ側推論 (run_edge) と 特徴量抽出
    E->>E: 分割点まで推論しつつ、<br/>クラウド側で必要なデータのみを選別保持
    end

    E->>E: INT8量子化 (イプシロンによるゼロ割防止)
    E->>C: TCP送信 (4バイト長ヘッダ + ペイロード)
    
    Note over C: 3. 受信 & データの復元 (Dequantize)
    Note over C: 4. クラウド側推論 (run_cloud)
    Note over C: 5. 後処理 (NMS: 重複検出の除去)
    
    C->>E: 検出結果の返送 (Pickleバイト列)
    Note over E: 6. 結果の可視化 (Resultsオブジェクト)
```
※Pickleは研究・ローカル環境用途を想定しており、信頼できない送信元に対しては安全ではない。

### 🛠 推論パイプラインの詳細
1. **[エッジ] 前処理 (Step 1)**: 入力画像を読み込み、`letterbox` を用いてYOLO入力サイズへの整形とテンソル化を行う。
2. **[エッジ] 推論実行と特徴量抽出 (Step 2)**: `run_edge` を実行し、`split_point` まで推論を進める。`needed` 集合（クラウド側で再利用されるレイヤー番号）に基づき、推論再開に必要な中間特徴量（`context`）のみを保持する。また、`last_use` によりエッジ内部で不要になった特徴量を即時解放し、メモリ使用量を削減する。
3. **[エッジ] 量子化・シリアライズ**: `edge_out` および `context` を INT8 に量子化する。ゼロ割防止のためイプシロン保護を適用し、メタデータを含むバイナリ形式へ変換する。
4. **[通信] 4バイト長ヘッダによるTCP通信**: TCPはメッセージ境界を保証しないため、先頭4バイトにデータ長を付与し、受信側で必要サイズを正確に取得している。
5. **[クラウド] 受信・復元・推論継続 (Step 3 & 4)**: 受信データを復元し、`run_cloud` により残りの計算グラフを実行する。
6. **[クラウド] 後処理 (Step 5)**: NMS（Non-Maximum Suppression）で重複検出を除去し、`scale_boxes` を用いて検出結果の座標を元画像サイズへ復元する。
7. **[エッジ] 結果の可視化 (Step 6)**: 返送された検出結果を元画像へ描画し、`ultralytics.engine.results.Results` オブジェクトを介して可視化する。

## 用語と主要概念 (Key Concepts)
本システムのコードを理解するための主要概念を以下にまとめる。
### 1. 分割推論のデータ制御
* **`run_edge` / `run_cloud`**  
  モデルを前後半に分割し、エッジ側・クラウド側でそれぞれ担当範囲を実行する関数。
* **`split_point`**  
  ネットワークを分割する層番号（インデックス）。
* **`edge_out` と `context`**  
  * **`edge_out`**: 分割層の直接出力テンソル。  
  * **`context`**: Skip Connection や Concat のためにクラウド側で再利用される中間特徴量。これがないと YOLOv8 の推論は正しく再開できない。
* **`needed` 集合**  
  クラウド側計算で必要となるレイヤー番号の集合。これを用いて送信すべき中間特徴量のみを選別する。
* **`last_use` (最終使用レイヤー判定)**  
  各特徴量がエッジ側で最後に利用される層を記録した辞書。推論中に不要となったテンソルを即時解放し、エッジ側メモリ使用量を削減する。

### 2. エンジニアリングの工夫
*   **LetterBoxを用いた前処理**:
    * アスペクト比を維持したまま画像を640×640へ整形し、物体の歪みを防ぐ。さらに、scale・pad情報を保持して検出座標を元画像へ正確に復元し、連続メモリ配置によりテンソル変換を高速化している。
*   **イプシロン(ε)によるゼロ割防止**: 
    *   データをINT8（256段階の整数）に圧縮（量子化）する際、最大値が0だと計算エラーが発生します。極小値（イプシロン）を足すことで、**どんな入力に対してもシステムをクラッシュさせない堅牢性**を確保しています。
*   **4バイト長ヘッダによるTCP通信**: 
    *   TCP通信は「データの切れ目」が保証されません。データの先頭に「今から何バイト送るか」という4バイトの情報を付与することで、**ストリーミング環境でも確実にデータを受信できる信頼性**を実装しています。
*   **NMS (Non-Maximum Suppression)**: 
    *   AIは1つの物体に対して複数の「検出枠」を出してしまうことが多いため、最も確率の高い1つに絞り込む数学的な後処理です。
*   **`ultralytics.engine.results.Results`**: 
    *   YOLOv8公式ライブラリが提供する標準的な出力形式。これを利用することで、検出結果の描画や保存を柔軟に行えます。

---

## 今後のマイルストーン (Future Milestones)

### Phase 1: 自動プロファイリングと最適化

* [ ] エッジ・クラウドにおける各層ごとの計算時間および通信時間を自動計測し、最適な split layer を探索するプロファイリング基盤の構築。
* [ ] 高性能 GPU 搭載クラウドサーバーへの移行と、複数エッジデバイスからの推論要求を継続的に受け付ける Edge–Cloud 推論サーバー基盤の構築。
* [ ] 前処理・推論実行・量子化/圧縮・通信処理を含むエンドツーエンド推論パイプラインの最適化。

### Phase 2: 動的環境への適応とスケーリング
* [ ] 研究室サーバー、大学スパコン、パブリッククラウド等、多様なノード環境への展開。
* [ ] 単一の静止画から、動画ストリーム（連続画像フレーム）を対象とした分割推論パイプラインの構築。
* [ ] 複数台のエッジデバイスが混在する環境や、動的に変動するネットワーク帯域下において、適切な分割点を自律的に模索・決定するアルゴリズムの実装。

### Phase 3: アプリケーション開発と社会実装 (Vision)
* [ ] **分散型AI監視システムの構築**: 小型エッジカメラから取得した映像を基に、人物認識や入退室管理をエッジ・クラウド協調でリアルタイムに実行。
* [ ] **自動省エネ管理システムへの統合**: 室内人数を継続的にトラッキングし、在室者ゼロを検知した際に空調や照明の消し忘れを判定。SlackやLINE APIと連携した自動通知・制御システムを実装し、スマートなエネルギー管理に貢献する。

---

# Edge-Cloud Collaborative Split Inference Optimization for YOLOv8
(Japanese version is above.)

## Project Overview

This project aims to develop a **split inference framework** that executes YOLOv8 inference collaboratively between **edge devices** (e.g., Raspberry Pi) and the **cloud** (or cloud servers), optimizing overall system latency and communication cost.

The goal is to realize a distributed AI system that efficiently coordinates the limited computational resources of edge environments with the high-performance computing power of cloud systems.

Rather than simply partitioning the model, this framework integrates:

* **Memory management considering Skip Connections**
* **Numerically stable quantization**

as its key technical contributions.

---

## Background & Challenges

With the rapid growth of IoT devices, advanced AI inference on edge devices has become increasingly important. However, a single edge device often lacks sufficient computational resources, while transmitting all raw data to the cloud introduces significant communication latency and bandwidth overhead.

This work adopts **Split Inference**, where a neural network is partitioned at a specific layer:

* the **front portion** (up to the split layer) runs on the edge
* the **remaining portion** runs on the cloud

The primary challenge is identifying the **optimal Split Point**, which minimizes the combined cost of:

* **Communication time**
* **Computation time**

under varying network conditions and heterogeneous computing resources.

---

## Methodology

This framework optimizes edge–cloud collaborative inference through two main design principles.

### 1. Measurement-Based Profiling and Optimization

* **Real-device profiling**  
  Instead of relying on simulation, computation and communication times are directly measured between physical devices at different split points to identify the configuration with minimum end-to-end latency.

* **Docker-based split inference**  
  Once the split point is selected, both edge and cloud execute inference within the same Docker environment for reproducibility and portability.

### 2. Maximizing Computational and Communication Efficiency

* **Inference mode (`torch.inference_mode()`)**  
  Gradient computation is disabled to reduce runtime overhead and memory usage during inference.

* **Intermediate feature selection and retention**  
  Considering YOLOv8 Skip Connections, only intermediate feature maps required by the cloud are retained and transmitted.

* **INT8 quantization**  
  Output tensors and intermediate features are compressed and reconstructed using INT8 quantization, significantly reducing communication volume.

## Current Status & Experimental Results

### 1. Prototype Environment

* **Edge**: Raspberry Pi 5 (16GB SDRAM, Docker environment)
* **Cloud**: MacBook Pro (8GB SDRAM, Docker environment)
* **Model**: YOLOv8n

---

### 2. Baseline Inference Performance

* **Raspberry Pi standalone**: 357 ms
* **MacBook standalone**: 131 ms

---

### 3. Split Inference Results

The following measurements were obtained for **YOLOv8n (22 internal Ultralytics modules)** using **Split Layer = 3**.

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Processing Time** | **762.8 ms** | End-to-end pipeline |
| **Preprocessing** | **284.6 ms** | Largest bottleneck |
| **Edge Computation** | 176.7 ms | Layers 0–3 |
| **Compression** | 4.8 ms | INT8 quantization + serialization |
| **Communication** | 46.4 ms | One-way approximation derived from TCP connection establishment and bidirectional transmission time |
| **Cloud Computation** | 203.9 ms | Layers 4–21 |
| **Visualization** | 567.9 ms | Excluded from total latency |
| **Communication Payload Size** | 402.44 KB | Approx. 8.47 MB/s effective throughput (≈ 68 Mbps) |

---

### 4. Analysis & Discussion

### Bottleneck Identification

Preprocessing (~285 ms) accounts for roughly **37% of total runtime**, making it the primary optimization target.

### Split Point Behavior

Split Layer **3–4** currently provides the best performance in this environment.

However, communication and Docker overhead still cause total latency to exceed Raspberry Pi standalone inference (357 ms).

### Quantization Effectiveness

Compression itself requires less than **5 ms**, indicating that INT8 quantization provides an efficient trade-off between:

* communication reduction
* inference practicality
* potential accuracy preservation

and remains a key factor for future deployment.

**[Split Inference Output Example]**

![Output by Split Inference](images/result_v1.jpg)

---

## Difference from Prior Work

A representative split inference study is Huawei's **Auto-Split**.

Auto-Split targets models such as ResNet and YOLOv3, selecting the split point that minimizes total latency through prior profiling of:

* Edge computation
* Communication
* Cloud computation

However, most prior work assumes relatively **sequential network architectures**.

YOLOv8, in contrast, includes:

* **Skip Connections**
* **Feature fusion through FPN / PAN**
* **Directed Acyclic Graph (DAG) structures with Concat operations**

FPN/PAN fuses fine-grained spatial information with high-level semantic information across multiple layers, enabling accurate detection of both small and large objects.

As a consequence, YOLOv8 reuses intermediate features generated in earlier layers during later cloud-side computation. Therefore, transmitting only the split-layer output is insufficient for correctly resuming inference.

To address this issue, this work implements:

* **`needed`**: a pre-analyzed set of layer indices reused on the cloud side
* **`last_use`**: tensor final-use tracking
* **`context` management**: retention of only required intermediate features

This enables **split inference compatible with YOLOv8's DAG architecture**.

Furthermore, while many prior studies estimate communication latency using theoretical bandwidth models, this work performs **measurement-based communication evaluation** including:

* Real socket communication
* INT8 quantization
* Serialization / Deserialization
* Tensor reconstruction

Therefore, this study extends traditional split inference for sequential networks toward an **edge–cloud inference framework designed for practical deployment**, explicitly considering both YOLOv8's complex feature fusion structure and real communication overhead.

## YOLOv8n Model Architecture

YOLOv8n consists of three major components:

* **Backbone** – extracts hierarchical visual features from the input image.
* **Neck** – fuses multi-scale features through FPN/PAN to improve detection performance for both small and large objects.
* **Head** – performs final object detection using multi-scale prediction features.

Unlike simple sequential CNNs such as ResNet, YOLOv8 adopts a **DAG (Directed Acyclic Graph)** architecture that incorporates **Skip Connections** and **Concat-based feature fusion**.

While this design significantly improves detection accuracy, it also introduces challenges for **Split Inference**, since features generated in earlier layers may be reused later in the network.

### YOLOv8n Layer Structure (Ultralytics Internal Modules)

```mermaid
flowchart TD

%% ===== Backbone =====
subgraph Backbone["Backbone (Feature Extraction)"]
    L0["0 Conv<br/>3→16"] --> L1["1 Conv<br/>16→32"]
    L1 --> L2["2 C2f<br/>32"]
    L2 --> L3["3 Conv<br/>32→64"]
    L3 --> L4["4 C2f<br/>64"]
    L4 --> L5["5 Conv<br/>64→128"]
    L5 --> L6["6 C2f<br/>128"]
    L6 --> L7["7 Conv<br/>128→256"]
    L7 --> L8["8 C2f<br/>256"]
    L8 --> L9["9 SPPF<br/>256"]
end

%% ===== Neck =====
subgraph Neck["Neck (FPN / PAN Feature Fusion)"]
    L9 --> L10["10 Upsample"]
    L10 --> L11["11 Concat"]
    L6 --> L11

    L11 --> L12["12 C2f"]

    L12 --> L13["13 Upsample"]
    L13 --> L14["14 Concat"]
    L4 --> L14

    L14 --> L15["15 C2f"]

    L15 --> L16["16 Conv"]
    L16 --> L17["17 Concat"]
    L12 --> L17

    L17 --> L18["18 C2f"]

    L18 --> L19["19 Conv"]
    L19 --> L20["20 Concat"]
    L9 --> L20

    L20 --> L21["21 C2f"]
end

%% ===== Head =====
subgraph Head["Head (Detection Head)"]
    L15 --> D["22 Detect"]
    L18 --> D
    L21 --> D
end

%% ===== Colors =====
style Backbone fill:#D6EAF8,stroke:#1F618D,stroke-width:2px
style Neck fill:#D5F5E3,stroke:#1E8449,stroke-width:2px
style Head fill:#FADBD8,stroke:#922B21,stroke-width:2px
```
*Note:* Numbers inside each module indicate the **layer index** and **channel dimensions**.  
`A→B` denotes **A input channels and B output channels**.

### Architecture Overview

| Component | Main Modules | Role |
|---|---|---|
| **Backbone** | Conv, C2f, SPPF | Extract hierarchical image features from low to high semantic levels |
| **Neck** | Upsample, Concat, C2f | Multi-scale feature fusion using FPN/PAN |
| **Head** | Detect | Final bounding-box and class prediction |

### Module Descriptions

| Module | Description |
|---|---|
| **Conv** | Basic convolution layer for feature extraction and channel transformation. |
| **C2f** | YOLOv8 feature block with skip connections. |
| **SPPF** | Pooling layer for efficient large-context aggregation. |
| **Upsample** | Increases feature-map resolution. |
| **Concat** | Concatenates feature maps along channel dimensions. |
| **Detect** | Detection head for bounding-box and class prediction. |

### Challenges of Split Inference in YOLOv8n

YOLOv8 is **not a purely sequential model**.  
Due to Skip Connections and Concat operations, features generated within the Backbone and Neck (e.g., Layers 4, 6, and 9) may be reused by later layers.

Therefore, in split inference, **transmitting only the output of the split layer is insufficient**.

For example, the following dependencies exist:

* **Layer11 ← Layer6**
* **Layer14 ← Layer4**
* **Layer17 ← Layer12**
* **Layer20 ← Layer9**

As a result, to correctly resume inference on the cloud side, it is necessary to transfer not only the split-layer output (`edge_out`) but also intermediate features (`context`) identified through dependency analysis.

The `needed`, `last_use`, and `context` mechanisms implemented in this work form the core technology enabling **Split Inference compatible with YOLOv8's DAG architecture** (details are described later).

---

## System Architecture & Flow

```mermaid
sequenceDiagram
    participant E as Edge (Raspberry Pi)
    participant C as Cloud (Mac / GPU Server)

    Note over E: 1. Image Loading & Preprocessing (LetterBox)

    rect rgb(240, 248, 255)
    Note over E: 2. Edge-side Inference (run_edge) & Feature Extraction
    E->>E: Run inference to split point<br/>while retaining only required features
    end

    E->>E: INT8 Quantization (epsilon-protected)
    E->>C: TCP Transfer (4-byte length header + payload)

    Note over C: 3. Receive & Restore (Dequantize)
    Note over C: 4. Cloud-side Inference (run_cloud)
    Note over C: 5. Post-processing (NMS)

    C->>E: Detection Result Return (Pickle byte stream)
    Note over E: 6. Visualization (Results object)
```

*Pickle is used for research and trusted local environments and is not secure against untrusted inputs.*

---

## Detailed Inference Pipeline

1. **[Edge] Preprocessing (Step 1)**  
   Input images are loaded and resized using `letterbox`, producing YOLO-compatible tensors.

2. **[Edge] Inference and Feature Extraction (Step 2)**  
   `run_edge` executes inference up to `split_point`. Based on the `needed` set, only intermediate features required for cloud-side continuation (`context`) are retained. `last_use` immediately releases unnecessary tensors to reduce edge memory usage.

3. **[Edge] Quantization and Serialization**  
   `edge_out` and `context` are quantized into INT8 format. Epsilon protection prevents division-by-zero during scaling, and metadata is serialized into binary form.

4. **[Communication] TCP with 4-byte Length Header**  
   Since TCP does not preserve message boundaries, a 4-byte header containing payload size is attached to ensure accurate reception.

5. **[Cloud] Reception, Reconstruction, and Inference Continuation (Step 3 & 4)**  
   Received data are reconstructed and `run_cloud` completes the remaining computation graph.

6. **[Cloud] Post-processing (Step 5)**  
   NMS removes duplicated detections, and `scale_boxes` restores coordinates to the original image resolution.

7. **[Edge] Visualization (Step 6)**  
   Returned detection results are rendered using `ultralytics.engine.results.Results`.

## Key Concepts

The following concepts are central to understanding the implementation of this system.

### 1. Data Control in Split Inference

* **`run_edge` / `run_cloud`**  
  Functions that divide the model into front and back segments, executing each portion on the edge and cloud respectively.

* **`split_point`**  
  The layer index where the network is partitioned.

* **`edge_out` and `context`**  
  * **`edge_out`**: Direct output tensor of the split layer.  
  * **`context`**: Intermediate feature maps reused on the cloud side due to Skip Connections or Concat operations. Without these tensors, YOLOv8 inference cannot correctly resume.

* **`needed` set**  
  A set of layer indices required for cloud-side computation. This enables selective transmission of only necessary intermediate features.

* **`last_use` (final-use layer tracking)**  
  A dictionary recording the last layer where each feature tensor is used. Tensors no longer required are immediately released, reducing edge memory consumption.

---

## Engineering Techniques

### LetterBox-based Preprocessing

Images are resized to **640×640** while preserving aspect ratio, preventing geometric distortion.

Additionally:

* scale and padding information are retained for accurate coordinate restoration
* contiguous memory layout accelerates tensor conversion and transfer

### Epsilon (ε) Protection Against Division-by-Zero

During INT8 quantization, scaling factors may become zero if tensor maximum values are zero.

By applying a small epsilon value:

* runtime errors are avoided
* system robustness is maintained for arbitrary inputs

ensuring stable quantization behavior.

### TCP Communication with a 4-byte Length Header

TCP does not preserve message boundaries.

Therefore, this framework prepends a **4-byte payload length header**, allowing the receiver to reconstruct complete messages reliably in streaming environments.

### NMS (Non-Maximum Suppression)

Object detectors often generate multiple bounding boxes for the same object.

NMS is a mathematical post-processing technique that retains only the highest-confidence prediction while suppressing redundant detections.

### `ultralytics.engine.results.Results`

A standard output representation provided by the YOLOv8 library.

Using this interface simplifies:

* visualization
* saving
* manipulation of detection results

---

## Future Milestones

## Phase 1: Automated Profiling & Optimization

* [ ] Build an automated profiling framework that measures per-layer computation and communication costs across edge and cloud environments to identify optimal split layers.
* [ ] Transition to GPU-enabled cloud servers and develop an Edge–Cloud inference server capable of continuously handling requests from multiple edge devices.
* [ ] Optimize the end-to-end inference pipeline including preprocessing, inference execution, quantization/compression, and communication.

---

## Phase 2: Adaptation to Dynamic Environments & Scaling

* [ ] Deploy the framework across heterogeneous environments such as laboratory servers, university supercomputers, and public cloud systems.
* [ ] Extend from single-image inference to continuous **video-stream split inference pipelines**.
* [ ] Develop adaptive algorithms that autonomously determine appropriate split points under:

  * dynamic network bandwidth
  * heterogeneous edge devices
  * changing system conditions

---

## Phase 3: Applications & Social Deployment Vision

### Distributed AI Surveillance System

* [ ] Develop a distributed surveillance platform using compact edge cameras, enabling real-time person detection and access monitoring through edge–cloud collaborative inference.

### Smart Energy Management Integration

* [ ] Continuously track room occupancy and detect unoccupied spaces to identify unnecessary lighting or HVAC operation. Integrate with Slack or LINE APIs for automated notification and control, contributing to intelligent energy management systems.