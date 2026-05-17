# pochishappy 利用ガイド

このドキュメントは, pochitrain で学習したモデルを `pochishappy` (CLI 名: `pochi`) で SHAP 可視化するまでの一連の手順を, チュートリアル形式で説明します.

CLI / config / 出力の **仕様書** は [README.md](../README.md) 側にあります.
本ドキュメントは「自分のプロジェクト向けに `pochi` を初めて動かす」ときの組み立て手順に焦点を当てます.

---

## 1. 前提

このガイドは以下の状態を想定しています.

- pochitrain で学習を完了し, `PochiModel.state_dict()` 互換の `.pth` ファイルを 1 つ手元に持っている
- pochitrain で使った **画像前処理** (mean / std / input_size / resize) を把握している
- pochitrain で使った **モデルアーキテクチャ** が `pochishappy` のサポート対象 (`resnet18`) に含まれる
- pochishappy のリポジトリを `uv sync` 済みで, `uv run pochi --help` が動く

学習時の前処理パラメータが分からない場合, この時点で pochitrain 側の config や学習スクリプトから値を控えておいてください.
これらが推論時の前処理と一致していないと, SHAP attribution は学習時とは別の画素空間で評価され, 可視化が意味を持ちにくくなります.

---

## 2. モデル準備

pochitrain の学習成果物から `.pth` を pochishappy 側に持ち込みます.

### 2.1 取り出すファイル

pochitrain の学習結果は実行ごとのディレクトリ (例: `work_dirs/<YYYYMMDD_NNN>/`) に保存され, `models/` 配下に以下のような `.pth` が生成されます.

```
work_dirs/<run_id>/
├── config.py             # 学習時の hyperparameter / 前処理
└── models/
    ├── best_epoch<N>.pth   # 検証スコアが最良だったエポックの state_dict
    └── last_model.pth      # 最終エポックの state_dict
```

このうち pochishappy に渡すのは **`best_epoch<N>.pth`** (または用途に応じて `last_model.pth`) です.
`pochishappy` は `torch.load(...)` で読み込んだ state_dict を `PochiModel` (= `torchvision.models.resnet18(num_classes=...)` 相当) にロードするため,
ファイルの中身は **`PochiModel.state_dict()` と同じキー構成** である必要があります.

(上記のディレクトリ名は pochitrain のバージョンや設定で変わることがあります. 各 run の `config.py` に学習時の前処理パラメータが残っているので, 4 章の config 作成時に併せて参照してください.)

### 2.2 配置例

利用者プロジェクト側のディレクトリ構成例 (README の「推奨ディレクトリ構成」と同じ):

```
your-project/
├── config.json
├── model.pth           # pochitrain の best.pth をコピー / リネーム
├── examples/
│   ├── data/
│   └── background/
└── out/
```

`.pth` をそのまま置く場合のおすすめ:

- ファイル名は `model.pth` のようにシンプルにする (config の `model` キーに書く名前と合わせる)
- 複数モデルを比較する場合は `model_v1.pth`, `model_v2.pth` のように suffix を付け, config 側を切り替える

---

## 3. 画像準備

`pochi` は 2 種類の画像セットを必要とします.

### 3.1 `--data` に渡す val 画像

SHAP で説明したい画像群です. `examples/data/` 配下に置きます.

- 対応拡張子: `.jpg`, `.jpeg`, `.png`, `.bmp`
- サブディレクトリは再帰探索されます (クラス別フォルダのままでも OK)
- 枚数は任意ですが, 1 枚あたりの SHAP 計算は秒〜分単位かかるため最初は 10〜30 枚程度で試すのが扱いやすい

```
examples/data/
├── cat/
│   ├── cat_001.jpg
│   └── cat_002.jpg
└── dog/
    ├── dog_001.jpg
    └── dog_002.jpg
```

### 3.2 `background_dir` に置く参照画像

SHAP の reference 分布 (= 「説明対象画像と比較するベースライン分布」) として使う画像群です.
`examples/background/` 配下に置きます.

選び方の指針:

- **学習データから抽出した代表サンプル 50 枚程度** を目安にする
  - 多すぎると SHAP の reference 計算コストが増える
  - 少なすぎると reference 分布が偏り attribution の安定性が落ちる
- **クラスバランス** を学習データに揃える (1 クラスに偏らせない)
- **val 画像と被らせない**. val と同じ画像を background に入れると, 説明対象画像と reference が一致する成分が出てしまい attribution がぼやける

---

## 4. config 作成

[`configs/example.json`](../configs/example.json) をコピーして自プロジェクト用に書き換えます.

```bash
cp configs/example.json ./config.json
```

`configs/example.json` は ImageNet 既定値 (mean / std) と ResNet18 224×224 の典型値が入った, そのまま動作するサンプルです.
書き換え時のチェックポイントは以下のとおりです.

### 4.1 学習時と一致させる必要があるキー

| キー | 役割 | 一致確認の方法 |
|---|---|---|
| `mean` | 学習時 dataloader の正規化 mean | pochitrain の dataloader / config から控える |
| `std` | 学習時 dataloader の正規化 std | 同上 |
| `input_size` | モデルに入る最終画像サイズ (正方形の一辺) | 学習時のモデル定義 (`PochiModel(input_size=...)` 相当) と揃える |
| `resize` | `CenterCrop` 前の中間リサイズサイズ | 学習時に `Resize(R) → CenterCrop(S)` を使っていたなら同じ `R` を入れる. 学習時に `Resize((S, S))` の直接リサイズだったなら `resize` キー自体を省略 |
| `arch` | torchvision アーキテクチャ名 | 学習時の `PochiModel` の backbone と一致させる. 現状 `"resnet18"` のみサポート |

`input_size` と `resize` の関係 (アスペクト比保持 vs. 強制スケール) は [README の該当節](../README.md#input_size-と-resize-の関係) に詳しく書いています.

### 4.2 自プロジェクトのパスに書き換えるキー

| キー | 書き換える内容 |
|---|---|
| `model` | 用意した `.pth` のパス. config からの相対パス (`./model.pth`) または絶対パスで書く |
| `background_dir` | 3.2 で用意した background 画像ディレクトリ (`./examples/background` など) |

### 4.3 任意で調整するキー

| キー | 用途 |
|---|---|
| `topk` | 1 つの画像で説明対象にする上位予測クラス数. 既定 `1`. 多クラス分類で top-K を比較したいときに増やす |
| `nsamples` | SHAP 推定試行回数. 既定 `20`. 大きいほど attribution が滑らかになるが速度が線形に悪化 |

### 4.4 書き換え後の config 例

```json
{
  "model": "./model.pth",
  "arch": "resnet18",
  "background_dir": "./examples/background",
  "mean": [0.4914, 0.4822, 0.4465],
  "std": [0.2470, 0.2435, 0.2616],
  "input_size": 224,
  "resize": 256,
  "topk": 1,
  "nsamples": 20
}
```

(上記は CIFAR-10 学習を想定した mean / std の例. 自プロジェクトの学習設定に合わせて差し替えてください.)

---

## 5. CLI 実行

config と画像が揃ったら `pochi` を実行します.

### 5.1 基本コマンド

```bash
uv run pochi \
  --config ./config.json \
  --data ./examples/data \
  --output ./out
```

CLI で一部だけ上書きしたい場合:

```bash
uv run pochi \
  --config ./config.json \
  --data ./examples/data \
  --output ./out \
  --topk 3 \
  --nsamples 50 \
  --device cuda
```

優先順位は **CLI 引数 > config > 既定値**. config 本体は触らずに試行錯誤したいときは CLI 上書きが便利です.

### 5.2 出力の見方

`--output ./out` の配下に以下が生成されます.

```
out/
├── predictions.csv
└── images/
    ├── cat_001.png
    ├── cat_002.png
    └── ...
```

- `images/<入力ファイル名 stem>.png` — SHAP 可視化図. `topk > 1` のときは top-K が同一図内に並びます
- `predictions.csv` — 画像ごとの予測結果サマリ. 1 行 = 1 入力画像で, 以下の列を持ちます

  | 列 | 内容 |
  |---|---|
  | `image_path` | 入力画像のパス |
  | `predicted_classes` | top-K の予測クラス index (JSON 文字列のリスト) |
  | `predicted_scores` | top-K の予測スコア (JSON 文字列のリスト) |
  | `shap_png` | 対応する SHAP 可視化 PNG のパス |

CSV の `predicted_classes` / `predicted_scores` は JSON 文字列なので, pandas で読む場合は `json.loads` を適用すると list として扱えます.

---

## 6. トラブルシュート

`pochi` 実行時によく起きる詰まり方と対処の目安です.

### 6.1 attribution が学習時と違って見える / 空間的に意味を持たない

**原因の候補**:

- `mean` / `std` が学習時と不一致 → 推論側で別空間の正規化がかかっており, モデルが期待する入力分布から外れる. attribution は数値的に出るが空間配置が崩れる
- `input_size` が学習時と不一致 → 受容野のスケールがずれ, attribution が想定外のスケールで広がる
- `resize` を省略しているが学習時は `Resize(R) → CenterCrop(S)` を使っていた (または逆) → アスペクト比の保持有無が逆転し, attribution が画像の歪んだ座標で評価される

**確認手順**:

1. pochitrain 側の学習 run の `config.py` (例: `work_dirs/<run_id>/config.py`) を開き, dataloader の正規化と Resize の値を控える
2. pochishappy の `config.json` と 1 つずつ突き合わせる
3. 特に `resize` キーは「指定なし = 強制スケール」「指定あり = アスペクト比保持 + CenterCrop」と挙動が変わる点に注意 (詳細は [README の該当節](../README.md#input_size-と-resize-の関係))

### 6.2 attribution がぼやける / コントラストが低い

**原因の候補**:

- `background_dir` の画像枚数が少なすぎる → reference 分布が偏り, SHAP 値の分散が暴れる
- `background_dir` の画像が val 画像と被っている → 説明対象画像と reference が一致する成分が出て attribution が薄まる
- `nsamples` が小さすぎる → サンプリングノイズで attribution が滑らかにならない

**対処**:

- background 画像を 50 枚程度に増やし, 学習データから多様にサンプルする
- val 画像と background を物理的に別ディレクトリに分け, 同じファイルが両方に入っていないか `diff` で確認する
- `--nsamples 50` などで試行回数を増やす (実行時間と引き換え)

### 6.3 終了コード `2` で落ちる

`pochi` は config 不正・入力パス欠落・値域エラーで終了コード `2` を返します.
エラーメッセージに該当キー名が出ます. 代表例:

- `config に必須キーが不足しています: ['background_dir']` → `config.json` のキー欠落. 4.1 / 4.2 のチェック表で確認
- `resize (200) は input_size (224) 以上で指定してください` → `resize >= input_size` の制約を満たしていない. 学習時の値で揃え直す
- `未サポートの arch: 'resnet50'` → 現状 `resnet18` のみサポート

### 6.4 終了コード `1` で落ちる (実行時エラー)

config は通ったが実行中に失敗した場合 (モデル読み込み失敗, CUDA OOM, 画像読み込み失敗など). 代表例:

- `.pth` の state_dict キーが `PochiModel` と合わない → pochitrain 側で `PochiModel.state_dict()` 互換の形で保存されているか確認 (例えば DataParallel ラップ後の `module.` prefix が付いていないか)
- CUDA OOM → `--device cpu` で動作確認, または GPU を変える
- 画像が壊れている / 拡張子だけ正しい不正ファイル → 該当ファイルを `examples/data/` から外して再実行

---

## 関連ドキュメント

- [README.md](../README.md) — CLI / config / 出力の仕様書, 推奨ディレクトリ構成
- [configs/example.json](../configs/example.json) — そのまま動作するサンプル config
