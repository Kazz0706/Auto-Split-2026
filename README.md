# Edge-Cloud Collaborative Split Inference Optimization for YOLOv8

## プロジェクト概要 (Project Overview)
本プロジェクトは、エッジデバイス（Raspberry Pi等）とクラウド（または高性能サーバー）間で物体検出モデル（YOLOv8）の推論処理を分割実行し、システム全体の処理効率を最大化する「動的分割推論フレームワーク」の構築を目的とする。
タイの国立研究機関（NSTDA）でのAIoT留学におけるスマートシステム開発の経験を原点とし、リソース制約の厳しいエッジ環境とクラウドの計算資源を最適に協調させる分散システムの実現を目指す。

## 背景と課題 (Background & Challenges)
IoTデバイスの普及に伴い、エッジ側での高度なAI推論が求められているが、単一のエッジデバイスでは計算資源が不足し、クラウドへの全データ転送は通信遅延や帯域幅の圧迫を招く。
本研究では、ニューラルネットワークの推論処理を特定の層で分割し、前半をエッジ、後半をクラウドで実行する「分割推論（Split Inference）」を採用する。
最大の課題は、ネットワーク状態や各ノードの計算能力に依存して変動する「通信時間」と「計算時間」の合計を最小化する**最適な分割点（Split Point）の動的特定**である。

## 手法とアプローチ (Methodology)
1. **実測ベースの最適化**: シミュレーションではなく、実機（Raspberry Pi, Mac等）間での計算時間および通信時間を実測し、最適な分割点を決定する。
2. **計算量・通信量の削減**: 
    * 中間表現（推論テンソル）から不要な勾配情報（`requires_grad=False`等）を除去し、計算およびメモリオーバーヘッドを削減。
    * モデルおよび転送テンソルの量子化（Quantization）を適用し、通信帯域の消費を抑制。

## 現在の進捗 (Current Status)
* **プロトタイプ構築**: Docker環境下において、Raspberry Pi（エッジ）とMacBook（サーバー想定）間でSocketを用いたTCP通信を実装し、YOLOv8の基本的な分割推論に成功。
* **データ転送の最適化**: 現在、ネットワーク転送ペイロードを純粋な推論データに最適化するため、送信フォーマットを`pickle`（メタ情報）と`.pt`ファイル（PyTorchテンソル）の組み合わせへ移行中。

## 今後のマイルストーン (Future Milestones)
### Phase 1: 基礎性能のプロファイリングと最適化
* [ ] Raspberry Pi - MacBook間における各層ごとの計算時間および通信時間の高精度な測定とプロファイリング。
* [ ] さらなる推論テンソルの圧縮（計算量削減）および量子化手法の適用（通信量削減）による、クラウド単体推論（Cloud-only Inference）に対する優位性の実証。

### Phase 2: 動的環境への適応とスケーリング
* [ ] 研究室サーバー、大学スパコン、パブリッククラウド等、多様なノード環境への展開。
* [ ] 単一の静止画から、動画ストリーム（連続画像フレーム）を対象とした分割推論パイプラインの構築。
* [ ] 複数台のエッジデバイスが混在する環境や、動的に変動するネットワーク帯域下において、適切な分割点を自律的に模索・決定するアルゴリズムの実装。

### Phase 3: アプリケーション開発と社会実装 (Vision)
*[ ] **分散型AI監視システムの構築**: 小型エッジカメラから取得した映像を基に、人物認識や入退室管理をエッジ・クラウド協調でリアルタイムに実行。
* [ ] **自動省エネ管理システムへの統合**: 室内人数を継続的にトラッキングし、在室者ゼロを検知した際に空調や照明の消し忘れを判定。SlackやLINE APIと連携した自動通知・制御システムを実装し、スマートなエネルギー管理に貢献する。

---

# Edge-Cloud Collaborative Split Inference Optimization for YOLOv8

## Project Overview
This project aims to build a **Dynamic Split Inference Framework** that maximizes overall system efficiency by dividing the inference process of the YOLOv8 object detection model between edge devices (e.g., Raspberry Pi) and the cloud (or high-performance servers). 
Inspired by my experience developing smart systems during an AIoT research program at the National Science and Technology Development Agency (NSTDA) in Thailand, this research seeks to optimally orchestrate heavily resource-constrained edge environments with cloud computing power to realize a robust distributed AI system.

## Background & Challenges
With the proliferation of IoT devices, there is a growing demand for advanced AI inference at the edge. However, relying solely on edge devices often leads to a shortage of computational resources, while offloading entire raw data to the cloud incurs significant communication latency and bandwidth congestion.
To address this, we adopt **Split Inference**, which partitions the neural network at a specific intermediate layer—executing the early layers on the edge and the remaining layers on the cloud.
The primary challenge is the **dynamic identification of the optimal split point** that minimizes the total sum of "communication time" and "computation time," which fluctuate depending on dynamic network conditions and the computing capabilities of each node.

## Methodology & Approach
1. **Empirical Measurement-Based Optimization**: Instead of relying solely on simulations, the optimal split point is determined based on real-world benchmarking of computation and transmission times across actual hardware (e.g., Raspberry Pi and Mac).
2. **Reduction of Computation & Communication Overhead**:
    * **Intermediate Tensor Optimization**: Removing unnecessary gradient information (e.g., using `requires_grad=False` or `torch.no_grad()`) from the intermediate representation to reduce computational load and memory footprint.
    * **Quantization**: Applying quantization techniques to both the model and the transferred tensors to significantly reduce communication bandwidth consumption.

## Current Status
* **Prototype Completed**: Successfully implemented basic split inference for YOLOv8 using TCP socket communication between a Raspberry Pi (Edge) and a MacBook (Server) within a Dockerized environment.
* **Data Transmission Optimization**: Currently migrating the network transmission payload to a combination of `pickle` (for metadata) and `.pt` files (for PyTorch tensors) to optimize the payload strictly for pure inference data.

## Future Milestones
### Phase 1: Baseline Profiling & Optimization
- [ ] Conduct high-precision measurements and profiling of computation and communication times per layer between the Raspberry Pi and MacBook.
- [ ] Demonstrate superiority over Cloud-only Inference through further compression of inference tensors (computation reduction) and quantization (communication reduction).

### Phase 2: Adaptation to Dynamic Environments & Scaling
-[ ] Deploy the framework across diverse node environments, including lab servers, university supercomputers, and public clouds.
- [ ] Expand the split inference pipeline to process continuous video streams rather than single static images.
- [ ] Implement an autonomous algorithm to dynamically search and determine the optimal split point under varying conditions, such as heterogeneous multi-edge environments and fluctuating network bandwidths.

### Phase 3: Application Development & Social Implementation (Vision)
- [ ] **Distributed AI Surveillance System**: Develop an end-to-end system that performs real-time person recognition and access control by orchestrating lightweight edge cameras with cloud resources.
- [ ] **Automated Energy Management Integration**: Implement a feature to continuously track room occupancy. Upon detecting zero occupancy, the system will identify forgotten air conditioning or lighting and trigger automated alerts/controls via Slack or LINE APIs, contributing to smart energy conservation.
