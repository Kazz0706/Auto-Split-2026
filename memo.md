# 手順
1.仮想環境 (venv) の作成と有効化, ライブラリのインストール
python3 -m venv venv　**作成 (.venv という名前のフォルダを作ります)**
source venv/bin/activate　**有効化 (Mac/Linux)**
pip install -r requirements.txt
(ラズパイの場合はDockerを使うのでvenv不要)

# Dockerコンテナ使い方
### Docker run
docker run -it \
  --name yolov8-cloud \
  -p 5001:5001 \
  -v $(pwd)/auto_split:/app/auto_split \
  <DockerHubのアカウント名>/yolov8-autosplit:py311 \
  /bin/bash
※-p 5001:5001でdockerコンテナとMacのポートを繋げる

## 停止中のコンテナを再起動
docker start <コンテナ名 or ID>
docker exec -it <コンテナ名 or ID> /bin/bash
1行でやるなら(前回プロセスに接続, bashじゃない可能性あり)
docker start -ai <コンテナ名 or ID>
## その他コマンド
1. docker exec ...「動いてるコンテナの中で新しいコマンドを実行する」
2. -it「ターミナル操作できるようにする」
    -i = interactive（標準入力を開く）
    -t = tty（疑似ターミナル）
例. docker exec -it my_container ls
    コンテナ内でls実行
3. -u root「rootユーザーとして実行」
権限エラー回避、apt installなどしたいため
4. 状態確認
docker ps        # 起動中
docker ps -a     # 全部

# Raspbery Pi 5について
## ラズパイからMacへファイル送信
scp [ラズパイのユーザ名]@[ラズパイのIPアドレス]:[ラズパイのファイルパス] [Macの保存先パス]
## Raspberry Pi 5 の注意点
最近のRaspberry Pi OS (Bookworm以降) は、システム全体への pip install を禁止しているため、venv の利用が必須。Dockerを使う場合は不要。
| 方法     | 役割         |
| venv   | Pythonだけ隔離 |
| Docker | OSごと隔離     |

# requirements.txt
ultralytics: YOLOv8を扱うための公式ライブラリ。
psutil: プロファイリング（メモリ使用量などの計測）に使う。
safetensors: 将来的に分割したデータを送信する際、pickleよりも高速で安全な形式として使う。

# Raspberry Pi & Macbook分割推論
## Docker Hub(+タグ管理)を使ってコンテナを共有
Docker Hub: アカウント名：kazumayoshida, パスワード：いつもの＋「@$」
### Macbook: Docker Hubにログインしてイメージをpush
1. docker login
2. docker tag yolov8-autosplit:py311 <yourname>/yolov8-autosplit:py311
(docker tag 元イメージ 新しい名前)
(Docker Hubは次の形式しか受け付けていないため   <ユーザー名>/<リポジトリ名>:タグ)
3. docker push <yourname>/yolov8-autosplit:py311
※ your nameはDocker Hubのアカウント名
dockerイメージは複数レイヤー(OS,Python,pip install,コード,追加ライブラリ,...)などで出来ているため、push時はレイヤーがいっぱい表示される
Dockerがレイヤー構造の理由：①高速化(変更があった部分だけ再push)②共有, 再利用性(Mounted from library/python)既にDocker Hubにあるレイヤーの使い回し③軽量
→ 変更があっても全部pushしていい。docker内部で変更されたレイヤーだけ自動で送るシステムがある。
### Raspberry Pi：Docker Hubからイメージをpullしてrun
1. Docker Hubからpullする
docker pull <yourname>/yolov8-autosplit:py311
2. Macbookからraspberry Piに必要なディレクトリ(COPY予定のディレクトリ)を送る(Docker build時にCPYし忘れたディレクトリについて)
scp -r auto_split_2026 yoshida@192.168.12.36:~
注：scpはコピー&転送なのでvolume mountは不可
3. yolov8-containerというコンテナ名にして保存したい場合(rmしない)
docker run -it \
    --name yolov8-container \
    -v $(pwd)/auto_split:/app/auto_split \
    -v $(pwd)/requirements.txt:/app/requirements.txt \
    <yourname>/yolov8-autosplit:py311 \
    /bin/bash
4. 再開
docker start -ai yolov8-container
5. 一覧
docker ps -a
6. 削除
docker rm yolov8-container

## Dockerコンテナをイメージにする(変更内容保存)
docker commit コンテナ名 イメージ名
注：commitでコンテナをイメージにすることが可能だが、ブラックボックス(Dockerfileに手順が残らない)なので避けるべき

## Dockerイメージの送信(非推奨, オフライン環境で有効)
注：コンテナ起動後に行った操作はイメージに反映されない → requirements.txtを再度ダウンロードする必要あり
　　イメージは固定, コードはmountできる
1. tarファイルにして保存：Docker Image → 1つのtarファイルに変換
docker save -o yolov8-autosplit.tar yolov8-autosplit:py311
2. ラズパイに送る
scp yolov8-autosplit.tar pi@<ラズパイのIP>:/home/pi/
3. ラズパイで復元
docker load -i yolov8-autosplit.tar
→ docker imagesで確認
4. ラズパイでコンテナ起動
docker run -it --rm yolov8-autosplit:py311
※さらに圧縮するなら
docker save yolov8-autosplit:py311 | gzip > autosplit.tar.gz
scp autosplit.tar.gz pi@IP:/home/pi/
gunzip autosplit.tar.gz
docker load < autosplit.tar
注：gzipは情報を1ビットも失わなわず、可逆圧縮である

# 複数デバイス環境
## multi-arch対応(amd, arm)でビルド
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t yourname/yolo:latest \
  --push .



# summary(model.model, input_size=(1, 3, 640, 640))の実行結果について

# Output Shape: [Batch, Channel, Height, Width]
その層でモデルを切断したときに **「通信しなければならないデータサイズ」**
Auto-Splitの目的は、このサイズが小さく、かつ計算が進んでいる場所を見つけること。

# 部品（レイヤー）の解説
Conv / Conv2d (Convolution): **「包丁」**です。画像をスキャンして特徴を切り出します。一番たくさん使われます。
BatchNorm2d: **「味見と調整」**です。データの数値が大きすぎたり小さすぎたりしないように、一定の範囲に整えます。
MaxPool2d: **「煮込み（濃縮）」**です。重要な情報だけ残して、画像のサイズを半分にします。データ量が減るので、Auto-Split的には「この直後」が通信のチャンスです。
Upsample: **「水増し（拡大）」**です。小さくなった画像を無理やり引き伸ばして大きくします。
Concat (Concatenate): **「合体」**です。2つの異なるデータをペタッとくっつけます。これがDAG（複雑な構造）の犯人です。
C2f(CSP Bottleneck with 2 convolutions): **「高性能な作業台セット」**です。YOLOv8特有の部品で、計算を軽くしつつ性能を上げるための工夫が詰まったブロックです。
SPPF(Spatial Pyramid Pooling - Fast): **「遠近両用メガネ」**です。画像の全体像と詳細を同時に見るための特殊なフィルタです。Backboneの最後に付いています。
Detect: **「検品係」**です。最後に物体がどこにあるかを決定する、Headの最終工程です。
DFL (Distribution Focal Loss): **「微調整テクニック」**です。箱の枠（Bounding Box）をより精密にするための数学的な工夫です。
ModuleList: 部品が入っている**「道具箱」**（プログラム用語）です。
Recursive: 「再帰」。同じ処理を繰り返す構造のことですが、ログに出ている場合は「中身が入れ子になっている」ことを指す場合が多いです。
Total params (5,257,936)
「脳みその重さ」。AIモデルの賢さです。約500万個の知識（パラメータ）を持っています。ファイルサイズでいうと約12MB程度です。
Total mult-adds (G) (4.37)
「計算の大変さ（汗の量）」。1回の推論で、約43億回（4.37 Giga）の掛け算・足し算が必要です。
Raspberry Pi 5にとって、この数値が処理時間（遅延）に直結します。
Input size (MB) (4.92)
入力画像（例えば640x640のカラー画像）がメモリ上で占めるサイズ。
Forward/backward pass size (MB) (232.15)
ここが最重要です。 「作業机の広さ」です。
計算の途中で生まれる「中間データ（切りかけの野菜）」を置いておくのに、約230MBのメモリを使います。
Raspberry Piのメモリが足りないと、ここでフリーズします（Pi 5なら余裕ですが）。
Trainable params: 0
「学習可能パラメータが0」。今は「推論（テスト）モード」なので、脳みそを書き換える機能はオフになっています、という意味です。

# ログの情報から分割点は...
1.Backboneで切る: SPPF: 2-51 より上の部分（Conv, C2f）のどこかで切断すれば、DAG（複雑な合流）を気にせず、単純にその時点の Output Shape のデータを送るだけで済みます。
2.通信量の見積もり:
Conv: 2-13 後で切ると 80x80x64 のデータ送信。
SPPF: 2-51 後で切ると 20x20x256 のデータ送信（こちらの方がだいぶ小さい）。
3.計算負荷の分散: SPPF までEdgeでやると、計算量（mult-adds）の半分近くをEdgeが背負うことになります。それが重すぎるなら、もっと手前の Conv: 2-3 などで切る、という調整を行います。

# ONNX
1.ノードの名前をクリック:画面右側に詳細が出ます。
2.name: これが Conv_13 や Concat_56 のような名前と対応します（※ONNX変換時に名前が少し変わる場合があるので、形状 Output Shape を頼りに照らし合わせます）。
Outputの確認:各ボックスの右側にある OUTPUTS の type を見てください。
float32[1,64,80,80] のような記述が見つかります。これが「ここで切った時の通信量」です。
3.BackboneとHeadの境界線:図をスクロールしていくと、一本道から急に矢印が飛び交うエリアに入ります。そこがHead（分割が難しいエリア）の入り口です。

# なぜHeadは複雑なのか？（FPN / PANet構造）
SPPF以降が複雑に見えるのは、単に予測するだけでなく**「情報の融合」**を行っているからです。
Backbone: 「これはタイヤ」「これは窓」というパーツを見つける。
SPPF: 「全体的に車っぽい」と理解する。
Head:
「車っぽい（SPPFの情報）」を拡大して、
「タイヤ（Backboneの途中の情報）」の場所と照らし合わせる（Concat）。
**「ここが車のタイヤだ！」**と特定する。



# run_autosplit関数について
1. if m_type == 'net': continue の意味
'net': YOLOの設定ファイル（.cfg）の一番最初には、必ず [net] というブロックがあります。これは 「モデル全体の設定」 を書く場所です。
画像のサイズ（width, height）, 学習率（learning_rate）, バッチサイズ
これらは「ニューラルネットワークの層（レイヤー）」ではなく、「設定情報」です。
依存関係マップを作るときに「この層は誰と繋がっているか？」と考える際、設定情報は邪魔なので無視 する必要があります。
continue の働き: continue は 「今回のループ（周回）はここで打ち切り、次のループへ進む（スキップする）」 という命令です。
2. consumers 関数の構造（データ構造）
consumers = {
    # キー(ID):  値(リスト)
    0: [1],          # 0番のデータは1番で使う
    1: [2],          # 1番のデータは2番で使う
    ...
    61: [62, 86],    # 61番のデータは62番と86番で使う (分岐！)
    ...
    86: [87]
}
3. module_defはmodel_defsの要素であり、辞書
model_defs 全体
model_defs = [
    {'type': 'net', 'width': 416, ...},                # module_def 0
    {'type': 'convolutional', 'filters': 32, ...},     # module_def 1
    {'type': 'shortcut', 'from': -3, ...},             # module_def 2
]
module_def が {'type': 'shortcut', 'from': -3} のとき
val = module_def['from']  # val には -3 が入る
4. route と shortcut の違い
| 項目 | **Shortcut** (ショートカット) | **Route** (ルート) |
| :--- | :--- | :--- |
| **計算方法** | **足し算 (+)** | **結合 (糊付け)** |
| **イメージ** | 重ねて値を混ぜる（絵の具を混ぜる） | 横や後ろにくっつける（ページを増やす） |
| **データの変化** | 変化なし (サイズもチャンネル数も同じ) | **チャンネル数（厚み）が増える** |
| **cfg内のキー** | `from` (どこから足す？) | `layers` (どこから持ってくる？) |
| **目的** | 学習をうまく進めるため (ResNet) | 複数の特徴を同時に使うため (FPN) |
コード上の違い:コード上では、どちらも「過去のレイヤーへの依存関係」を生むので、consumers マップを作る上での処理は似ていますが、cfgファイル内で使われているキーワード（from か layers か）が違うため、別の if 文で処理しています。


# 機械学習知識
## OpenCVとは
Open Source Computer Vision Library
用途：画像読み込み, 動画処理, フィルタ, 幾何変換, 特徴点検出, 物体検出（古典手法）
メリット：画像IOが便利, 高速, 安定


# image_autosplit
## 座標変換の流れ：
orig image (H0×W0)
        ↓ scale + pad
model input (640×640)
        ↓ 推論
pred boxes (640座標)
        ↓ scale_boxes
orig座標に復元
## 行列の流れ: 勾配爆発を防ぐため 0~255 → 0~1.0
JPEG
 ↓
img0 (H,W,3) uint8 0~255 (H,W,3)
 ↓ LetterBox
img uint8 640×640
 ↓ /255 normalize
float 0〜1
 ↓ CHW + batch
(1,3,640,640)
 ↓
YOLO
 ↓
pred (1,84,8400)
 ↓ NMS
det (N,6)
 ↓ scale_boxes
orig座標
 ↓ plot
orig_img + box
## UltralyticsのLetterBoxとscale_boxesについて
scale_boxes ≈ LetterBox の逆写像
### LetterBoxの中身: スケール + パディング
変換: 元の画像(H0, W0)→スケール後の画像(H1, W1) ※Hは高さ, Wは横幅
scale = min(640 / H0, 640 / W0)
H1 = round(H0 * scale)
W1 = round(W0 * scale)
その後Padding: # 片方は0になり、もう片方は余白を表す→正方形のキャンバスに長方形の画像を収める感じ。Ultraliticsの仕様で余白は0 ではなく 114（灰色）で埋められる
pad_x = (640 − W1) / 2
pad_y = (640 − H1) / 2
処理後の画像
x_model = x_orig * scale + pad_x
y_model = y_orig * scale + pad_y
最終画像: 
img ∈ R^(640 × 640 × 3)
### scale_boxesの中身: unpad + unscale
x_orig = (x_model − pad_x) / scale
y_orig = (y_model − pad_y) / scale
## その他疑問
### org_img→img変換の過程で、255で割った値はいつ戻す？
戻さない。NNは0~1正規化を前提に学習される。
### 出力テンソル(1,84,8400)の中身
batch = 1
84 = 4(box) + 80(class)
8400 = 候補数

4 (box): [x(boxの中心のX座標), y(中心Y座標), w(画像の幅), h(画像の高さ)]
80(class): [class1, class2, ..., class80] -> クラスごとの確率を格納(クラスはCOCO datasetに基づく)

8400(候補数): 各セルが「物体があるかも」という候補
80×80 = 6400
40×40 = 1600
20×20 = 400
--------------
合計 = 8400
マルチスケール検出
| スケール | 特徴マップ | stride | 1セル対応 | 物体 |
| ---- | ----- | ------ | -------- |
| P3   | 80×80 | 8      | 8×8 px   | 小物体 |
| P4   | 40×40 | 16     | 16×16 px | 中 |
| P5   | 20×20 | 32     | 32×32 px | 大 |
※通常のCNNでいうstrideは「カーネルを何ピクセルずつ動かすか」だが、YOLO検出ヘッドでは「stride = 入力画像に対する縮小倍率」ネットワーク全体の累積縮小率

8400の各セル: 「この領域(セル)に物体があるなら、その中心はここらへんで、サイズはこれくらい、クラスはこれっぽい」というboxとclassを紐付けた確率情報
→3スケールで縮小したものは、最後のscale_boxes()の1回だけで戻している

# search.py
## torch.no_grad()とtorch.inference_mode()の違い
with torch.no_grad():勾配計算をしない。微分計算システムAutogradの仕組み自体は生きている。計算グラフは作らないが、Tensorは通常モード。
with torch.inference_mode():推論専用モード(完全にAutogradを無効化)。version counter を止める。view tracking 無効化。inplace安全チェックを一部スキップ。バグの検出が一部飛ばされるので安全性が少し下がる。しかし、x = m(x_in)だけならinplace操作もview依存もないので問題なし。

version counter：バグ検出用の更新回数カウンタ
view tracking：同じメモリを共有
inplace安全チェック：書き換えは元の値を破壊するのでgradient計算と衝突する可能性をチェックしてる。

| 項目         | no_grad | inference_mode |
| ---------- | ------- | -------------- |
| 勾配計算       | しない     | しない            |
| Autograd内部 | 生きてる    | 完全停止           |
| 速度         | 速い      | **さらに速い**      |
| メモリ        | 削減      | **さらに削減**      |
| 安全性        | 高い      | やや制限あり         |

## その他疑問
### Pythonのwith：一時的に状態を切り替える
with A():
    処理
は内部的に以下の処理
A.__enter__()
処理
A.__exit__()

### self.model.eval()：レイヤーの振る舞いを変える
BatchNorm / Dropout の挙動を推論モードにする
Dropout：trainモードの時はランダムにニューロン消すが、evalモードの時は何も消さない
BatchNorm：バッチ正規化y=​x−μ/√σ^2+ϵ (μ = 平均, σ² = 分散)
trainモード時はその時の入力(バッチ)のu, σを使う。evalモード時は学習時に蓄積されたu, σを使う。 → eval()を使わないと毎回結果が変わる。
補足：学習時に蓄積した平均, 分散はwrapper.yolo.model内の各BatchNormに入っている。
trainモードの危険性：バッチサイズ1のとき、例えば、バッチ[10], 平均10, 分散0で正規化すると0になる → 特徴が全部壊れる。YOLOでは小さい物体が消える。→バッチサイズが小さいと危ない。
yolov8n.pt= 重み + BatchNormの統計

### 分割推論で枠がズレる原因top3
① BatchNormの統計ズレ ② 量子化誤差 ③ scale / padの不整合

### BatchNorm：単なる式ではなく、状態（パラメータ）を持つレイヤー
🔹 BatchNormの中身（実体）
各 BatchNorm2d はこういう変数を持っています：
m.weight        # γ（スケール）
m.bias          # β（シフト）
m.running_mean  # 学習で蓄積された平均
m.running_var   # 学習で蓄積された分散
🔸 どうやって「蓄積」されるか
学習中（trainモード）：
running_mean = 0.9 * running_mean + 0.1 * μ_batch
running_var  = 0.9 * running_var  + 0.1 * σ²_batch
👉 指数移動平均（EMA）で更新

YOLOの場合
wrapper.yolo.modelの中にある全てのBatchNormが「COCOデータで学習した統計」を持っている。つまりyolov8n.pt = 重み + BN統計

### version counterは何のため？
🔹 正体
Tensorは内部にversion_counterを持ってる。
🔸 役割
「このTensorが途中で書き換えられてないか？」を監視する。
🔸 例
x = torch.ones(3)
y = x * 2   # yはxに依存
x += 1      # xを書き換え
このとき：yの計算に使ったxが変わった → 不整合
👉 version counterで検出
※推論では基本いらない。理由：backpropしない, 履歴追わない👉 inference_modeでOFFにされる

### view trackingは何のため？
🔹 viewとは...y = x.view(-1)これは同じメモリを別の形で見る
🔸 危険なケース
x = torch.ones(4)
y = x.view(2,2)
y[0,0] = 999
👉 xも変わる
x = [999, 1, 1, 1]
🔹 view trackingの役割：「どのTensorが同じメモリを共有してるか」を追跡
分割推論ではほぼ不要。理由：forward onlyでinplaceしない, grad使わない

### inplaceとgradient衝突の具体例
🔹 危険なコード
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
x += 1   # inplace変更
z = y.sum()
z.backward()
y = x * 2 の時点でy は「古い x」を使っている。でもその後：x が書き換えられる
計算グラフの矛盾：y = 2x_oldでも backward は x_new を使う？👉 矛盾

### inference_modeは「速いが、Tensorを後処理でいじると壊れる」
YOLOでよくある落とし穴
NMS, cale_boxes, postprocess ← 全てinplaceあり👉 全部アウト

### PyTorchのTensorは以下の情報をもつ配列
device（CPU/GPU）, dtype（float32など）, requires_grad, view関係, inference_modeフラグ

### detach()
履歴を捨てて出力の数値だけ残す
計算グラフ(backprop用)を切る, Autogradから切り離す, データは共有（コピーしない）
inferenceモードで最初からgraphを作らないが、念の為途中でgraphを切るdetachを使用

### clone()
view tracking, counterなど不足している情報は初期状態で作られる
メモリごとコピー, 完全な別Tensor
cloneは inference_modeの外で実行され、新しいTensor(通常Tensor)を作る

# edge.py
## socket通信
### header = struct.pack(">I", len(data))
1. struct：バイナリ変換ライブラリ
2. pack：Pythonの値 → バイト列に変換
3. ">I"：フォーマット指定
    >	ビッグエンディアン（ネットワーク標準）
    I	unsigned int（4バイト整数）
全体の意味：「データ長（len(data)）を4バイト整数に変換」
必要な理由：TCPはデータの区切りがない👉「どこまでが1メッセージか分からない」👉最初にサイズ送る
※TCP (Transmission Control Protocol): 信頼性と順序保証を重視（Web, メール等, SSH）
UDP (User Datagram Protocol): 即時性と低遅延を重視（動画配信, 音声通話, オンラインゲーム等）
### PORTについて
with socket.socket(...) as s:
👉スコープ終了時に close() 実行 → PORT 5001が自動で閉じる
🔥 注意：同時に複数起動すると競合する
socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:以下
1. socket.socket()：ソケット作成（通信口）
2. socket.AF_INET：IPv4通信
※ IPv4（Internet Protocol Version 4）：2^32(約43億)個の固有ID（IPアドレス：192.168.x.x等）を割り当てる従来の通信規格
3. socket.SOCK_STREAM：TCP通信
🔥 つまり「IPv4 + TCP の通信を作る」
4. with socket.socket(...) as s：作ったソケットを s という変数で使う＋自動closeされる
5. s.connect((HOST, PORT))：指定したIPとポートに接続。接続を確立する（3-way handshake）
6. s.sendall(header + data)
接続先（HOST:PORT）に[4byte: データサイズ][本体データ]を送信
🔥 重要：TCPは「区切りがない」ので、自分で区切りを作ってる
    sendall = 全部送るまでブロック
❌ sendとの違い：s.send(...)👉途中で終わる可能性あり

# cloud.py(先に起動)
## HOST = "0.0.0.0" 
「すべてのネットワークインターフェースで待つ」サーバーは基本これ
## def recvall(sock, n):
「nバイト必ず受け取る」必要な理由：recv()は途中で切れる
1. s.bind((HOST, PORT))「このIPとポートを自分が使う」
2. s.listen(1)：接続待ち状態に入る
    引数 1　→ 同時接続数（バックログ）

3. conn, addr = s.accept()：接続が来るまでブロック（待機）
conn	通信用ソケット
addr	相手のIP
接続後 with conn: 👉 通信専用ソケット

header受信：header = recvall(conn, 4)
    4バイト受信（データサイズ）
サイズ復元：msg_len = struct.unpack(">I", header)[0]
    バイト → 整数（データ長）
本体受信：data = recvall(conn, msg_len)
    データ本体を全部受信
デコード：packet = pickle.loads(data)
    バイト → Pythonオブジェクト

## 全体の流れ（完全理解）
[Cloud]
bind → listen → accept（待つ）
        ↓
[Edge]
connect → send
        ↓
[Cloud]
header受信 → サイズ取得 → 本体受信

## withとは「開始時に準備して、終了時に必ず後処理する仕組み」
with A as x:
    処理

👉 内部的には
x = A.__enter__() # 入るときenter
try:
    処理
finally:
    A.__exit__() # 出るときexit（必ず実行）
例① with socket.socket(...) as s:
    終了時に s.close() が自動実行
例② with open(...) as f:
    終了時に f.close()
例③ with torch.inference_mode():
    内部フラグをON → 終了時に元に戻す
✔️ 共通点「リソース管理（開く→閉じる）」を安全にやる
## ソケットとは？
ソケット = ネットワーク通信の「端点（口）」
構成：(IPアドレス, ポート番号)
例：192.168.1.10:5001
🔥 役割
サーバ	bind + listen + accept
クライアント	connect
## pickle.loadsとは？
バイト列 → Pythonオブジェクトに復元
data = pickle.dumps(obj)   # → バイト
obj = pickle.loads(data)   # → 元に戻す
🔥 注意：構造（dict, list, tensorなど）を丸ごと復元
④ struct.unpack(...)[0] の意味
msg_len = struct.unpack(">I", header)[0]
✔️ unpackの戻り値
struct.unpack(">I", header)👉(12345,)   # タプル
なぜタプル？：複数値対応のため
✔️ [0]の意味：タプルの1番目を取り出す
❗ そのままだと：msg_len = (12345,)
## data = b"" の b とは？
b = バイト列（bytes型）
例：b"abc"  # bytes = バイナリデータ（通信・画像など）
"abc"   # str
🔥 socketはbytesしか扱えない
## sock.recv の挙動
「最大nバイト受け取る（保証ではない）」
🔥 重要：少ししか来ないこともある, 0バイト → 接続終了
packet = sock.recv(n - len(data))で「残り必要な分だけ受け取る」
## recvallの全体像
data = b""
while len(data) < n:
    packet = sock.recv(n - len(data))
    data += packet
✔️ 意味
小分けに届くデータを全部つなげる

## サーバー用ソケットと通信用ソケットの違い
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:はサーバー用
with conn:は通信用
サーバー用の中にlisten(n)ならn個の通信用ソケットwith conn:が作れる
| 変数     | 役割                 |
| ------ | ------------------ |
| `s`    | サーバソケット（待ち受け専用）    |
| `conn` | 通信ソケット（recv/send用） |


・tensor_buffer = io.BytesIO()
メモリ上の仮想ファイルを作る。BytesIOはRAM上の .pt ファイル
普通はtorch.save(tensor, "file.pt")ディスクに書き込む
→しかしそれだと、ディスクI/O（遅い）、ファイル管理が必要なのでRAM上にファイルを作る方法を使っています。
・torch.save(edge_out_q.cpu(), tensor_buffer)
Tensor → PyTorch形式バイナリ
edge_out_q → .pt形式 → tensor_buffer (RAM)
GPU Tensor → CPU Tensorに変換。GPUメモリはpickleやsocket送信できないため
・tensor_bytes = tensor_buffer.getvalue()

BytesIOに保存されたデータをbytes型として取り出す。つまり.ptファイル（RAM上）→ raw bytes


meta_bytes = json.dumps(meta).encode()
metaはPythonのdict。
meta = {
 "scale": 0.75,
 "pad": (0, 40),
 "orig_shape": (1080,1920)
}
これをJSON化：json.dumps(meta)
'{"scale":0.75,"pad":[0,40],"orig_shape":[1080,1920]}'
bytes化：.encode()
b'{"scale":0.75,"pad":[0,40],"orig_shape":[1080,1920]}'
ソケット通信はbytesしか送れないため

packet = {
    "tensor": tensor_bytes,
    "meta": meta_bytes,
    "scale_q": scale_q.cpu().numpy(),
    "split": split_point
}
packet
 ├ tensor → YOLO中間特徴量
 ├ meta → LetterBox情報
 ├ scale_q → 量子化復元用
 └ split → 分割位置

scale_q.cpu().numpy()
Tensor → numpyに変換するとpickleが軽くなる
Tensorのままだとdevice, dtype, grad, storageなどが保存されてしまう。

### tensor.save() vs pickle
pickle＝汎用Pythonシリアライザ
Tensor → pickleすると内部で
Tensor
 ├ dtype
 ├ device
 ├ requires_grad
 ├ storage
 ├ stride
 ├ layout
 └ metadata
など大量の情報を処理します。加えて、Pythonオブジェクトとして再帰的にシリアライズ→遅い

torch.saveはTensor専用なので
Tensor
 ├ dtype
 ├ shape
 └ raw memory
をC++で高速バイナリ保存



結論から言うと、**研究として綺麗なのは「完成後にアブレーションスタディで比較」**です。
ただし、**metaの扱いは今のコードだと確かに冗長**なので整理した方がいいです。順番に説明します。

---

# ① torch.save vs pickle の比較はいつやるべきか

あなたの研究の場合は **後でやるのが正しい**です。

理由は研究の構造として

```
① システム完成
② 最適化
③ アブレーション
```

が基本だからです。

今やると

```
torch.save vs pickle
pickle vs json
socket vs file
```

などが途中で増えてしまい、

**実験の軸がブレます。**

---

## 論文や修論の構成

普通はこう書きます

### Method

提案手法

```
Tensor → torch.save
meta → json
INT8 quantization
```

---

### Experiment

Baseline

```
pickle serialization
```

Proposed

```
torch.save serialization
```

---

### Ablation Study

```
serialization
quantization
split position
```

例えば

| Method            | Latency |
| ----------------- | ------- |
| pickle            | 120 ms  |
| torch.save        | 95 ms   |
| torch.save + INT8 | 70 ms   |

---

つまり

**実装は先に完成させるべき。**

その後

```
pickle版
torch.save版
```

を比較します。

---

# ② 今やるべきこと

あなたの研究の優先順位は

```
① Edge↔Cloud通信完成
② latency測定
③ split探索
④ 最適化
⑤ ablation
```

なので

**serialization比較は④か⑤**

です。

---

# ③ metaがcloudに必要か

これは良い指摘です。

結論

```
run_cloudには不要
```

です。

あなたの理解は正しい。

---

### run_cloudの役割

```
Edge feature
↓
残りのYOLO layer
↓
Detection tensor
```

ここでは

```
画像サイズ
scale
padding
```

は一切使いません。

つまり

```
meta不要
```

---

# ④ metaが必要な場所

metaが必要なのはここです

```
scale_boxes()
```

ここ。

```
det[:, :4] = scale_boxes(
    meta["input_shape"],
    det[:, :4],
    meta["orig_shape"]
)
```

理由

YOLOは

```
640×640座標
```

で出力するからです。

元画像が

```
1920×1080
```

なら

```
座標復元
```

が必要。

---

# ⑤ つまり役割はこう

Edge

```
LetterBox
↓
meta作成
```

Cloud

```
推論
↓
NMS
↓
scale_boxes(meta)
```

---

# ⑥ 今のコードの問題

今は

```
run_cloud(x, context, split, meta)
```

になっていますが

**metaは使っていない。**

なので

正しくは

```
run_cloud(x, context, split)
```

です。

---

# ⑦ 正しい構造

---

### 呼び出し

```python
final_result = wrapper.run_cloud(edge_out, context, split_point)
```

---

### metaはここで使う

```python
det[:, :4] = scale_boxes(
    meta["input_shape"],
    det[:, :4],
    meta["orig_shape"]
)
```

---

# ⑧ 研究的にはもっと重要なポイント

実はあなたの研究で一番重要なのはここです

```
context を送る必要があるか？
```

YOLOは

```
skip connection
```

があるので

```
saved_y
```

を送る必要があります。

しかし

**これが通信量を増やします。**

なので論文では

```
splitはskipの後にする
```

ことが多い。

---

# ⑨ 研究として綺麗な構造

あなたの研究をまとめると

```
Edge
画像
↓
LetterBox
↓
Edge layers
↓
INT8量子化
↓
torch.save
↓
socket

Cloud
socket受信
↓
torch.load
↓
Cloud layers
↓
NMS
↓
scale_boxes
```

metaは

```
scale_boxes専用
```

です。

---

# ⑩ 研究的に一番危険なポイント

今あなたのコードには

**もう一つ大きな研究ポイントがあります。**

それは

```
context (skip connection)
```

です。

YOLOv8nの場合

```
split位置によって
通信量が10倍変わる
```

可能性があります。

これは

**AutoSplit論文の核心部分**

です。

---
self.layersは以下の右の要素(nn.Module)を並べた全レイヤーのリスト(nn.ModuleList)
self.layers
│
├ layer0  Conv
├ layer1  C2f
├ layer2  Conv
├ layer3  C2f
├ layer4  Conv
├ layer5  C2f
├ layer6  SPPF
├ layer7  Upsample
├ layer8  Concat
├ layer9  C2f
├ layer10 Upsample
├ layer11 Concat
├ layer12 C2f
├ layer13 Detect

x = m(x_in)はニューラルネット1層のforward計算
例）x = Conv(x) は x = W * x + b を計算
    x = C2f(x)はconv→bottleneck→concat→conv
    x = Concat([x1, x2])は 内部でtorch.cat([x1, x2], dim=1)
m.fはYOLO特有の接続情報(from)
m.fが1入力ならConv, C2f, SPPF, Unsample, Detectなど普通のNN層
m.fが複数入力ならAdd, Mul, Attention, Sumなど2つの要素を用いた計算をする層
-> ResNetならAdd, YOLOv8の構造ではConcatしか使われない
※YOLOv8は一つのブロック(レイヤー)内でAddやMulは使われることはあっても、ブロック単位ではconcatまたは単体計算になる
ブロック内部(Conv, C2fなど) → Add / Mul あり
ブロック接続(YOLOのレイヤー接続) → 単入力 or Concat

## 描画の仕方
方法①Ultralyticsを使う
from ultralytics.engine.results import Results
r = Results(
    orig_img=orig_img,
    path="img",
    names=wrapper.yolo.names,
    boxes=torch.tensor(boxes)
)
plotted = r.plot()
cv2.imwrite("result.jpg", plotted)
👉 これでラベル表示, 信頼度表示, 色付き, 太さ自動
方法②OpenCVで自作描画（軽量・研究向け）
for box in boxes:
    x1,y1,x2,y2,conf,cls = box
    label = f"{wrapper.yolo.names[int(cls)]} {conf:.2f}"
    cv2.rectangle(orig_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
    cv2.putText(
        orig_img,
        label,
        (int(x1), int(y1)-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )