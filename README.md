# pochishappy
Make your SHAP analysis happy — Pochi explains it all with a wagging tail!

## MVP

`pochishappy` は画像分類モデル向けの SHAP 可視化を生成する CLI ツールです.
学習済みの `.pth` モデルと val 画像を入力に, `shap.GradientExplainer` で SHAP 値を計算し PNG を出力します.

## 入出力契約

### CLI

```bash
uv run pochi \
  --config ./config.json \
  --data ./val \
  --output ./out
```

| 引数 | 必須 | 意味 |
|---|---|---|
| `--config` | ✓ | JSON config ファイルのパス |
| `--data` | ✓ | 入力画像ディレクトリ. サブディレクトリを再帰探索する |
| `--output` | ✓ | 出力先ディレクトリ |
| `--model` | ✗ | config の `model` を上書きする `.pth` パス |
| `--background-dir` | ✗ | config の `background_dir` を上書きする画像ディレクトリ |
| `--topk` | ✗ | config の `topk` を上書きする上位予測クラス数 |
| `--nsamples` | ✗ | config の `nsamples` を上書きする SHAP 推定試行回数 |
| `--device` | ✗ | 推論デバイス (`auto` / `cpu` / `cuda`). 既定: `auto` |

優先順位は **CLI 引数 > config > 既定値**.

### `config.json`

```json
{
  "model": "./model.pth",
  "arch": "resnet18",
  "background_dir": "./train_sample",
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "input_size": 224,
  "resize": 256,
  "topk": 1,
  "nsamples": 20
}
```

| キー | 型 | 必須 | 意味 |
|---|---|---|---|
| `model` | str (path) | ✓ | `.pth` モデルファイルのパス. config からの相対 or 絶対 |
| `arch` | str | ✓ | torchvision アーキテクチャ名. 現状 `"resnet18"` のみサポート |
| `background_dir` | str (path) | ✓ | SHAP の reference 分布になる画像群のディレクトリ. 学習データから抽出した代表サンプルを置く |
| `mean` | list[float] (3 要素) | ✓ | 学習時の正規化 mean |
| `std` | list[float] (3 要素) | ✓ | 学習時の正規化 std |
| `input_size` | int | ✓ | モデル入力の最終サイズ. 学習時と一致させる |
| `resize` | int | ✗ | 中間リサイズサイズ. 指定時は `Resize(resize) → CenterCrop(input_size)`, 省略時は `Resize((input_size, input_size))` 直接リサイズ |
| `topk` | int | ✗ | 説明対象にする上位予測クラス数. 既定 `1` |
| `nsamples` | int | ✗ | SHAP 推定試行回数. 既定 `20`. 大きいほど精度向上, 速度低下 |

`mean` / `std` / `background_dir` / `input_size` などは学習時の前処理と一致させる必要があります. 整合しない値で実行すると attribution が誤った可視化になります.

#### `input_size` と `resize` の関係

`input_size` は **モデルに入る最終的な画像サイズ** (正方形の一辺) で, 学習時のネットワーク入力と一致させる必要があります.
`resize` は **`CenterCrop` 前の中間リサイズサイズ** で, 任意指定です. 二者の関係で前処理パイプラインが切り替わります.

| `resize` 指定 | 適用される変換 | アスペクト比 | 用途 |
|---|---|---|---|
| 指定あり (`resize >= input_size`) | `Resize(resize) → CenterCrop(input_size)` | **保持** (resize 段階) → 中央を crop | 学習時に `Resize → CenterCrop` を使ったモデル. ImageNet 典型: `resize=256, input_size=224` |
| 省略 (`null` / キーなし) | `Resize((input_size, input_size))` | **崩れる** (両辺強制スケール) | 学習時にアスペクト比を捨てて直接リサイズしたモデル. 入力が非正方形でも歪めて正方形化する |

torchvision の `transforms.Resize` は引数の型で挙動が変わります.

- `Resize(256)` (int 1 つ): **短辺**を 256 にスケールし, 長辺は元の比率で連動. 例えば 640×480 の画像は 341×256 になる. アスペクト比は保持される.
- `Resize((224, 224))` (タプル): 両辺を強制的に 224 にする. 640×480 の画像も 224×224 に縮められ, アスペクト比は崩れる.

したがって `resize=256, input_size=224` を指定した場合の流れは:

1. `Resize(256)`: 短辺を 256 に揃える (アスペクト比保持. 例: 640×480 → 341×256)
2. `CenterCrop(224)`: 中央 224×224 を切り出す (最終出力は正方形だが「歪み」ではなく「視野の切り取り」)

省略時の `Resize((224, 224))` は長方形画像を強制的に正方形へ歪ませる挙動なので, 学習時にこのスタイルを使ったモデル以外では SHAP attribution が空間的に意味を持ちにくくなります.

制約:

- `resize` は `input_size` 以上である必要があります. `resize < input_size` の場合, config ロード時に `ValueError` で終了コード `2` を返します.
- 「学習時の前処理と一致させる」のが大原則です. 学習側で `Resize(256) → CenterCrop(224)` を使っていたなら推論側も同じ組み合わせにする (`resize=256, input_size=224`) ことで, SHAP attribution が学習時と同じ画素位置で評価されます.

対応画像拡張子は `.jpg`, `.jpeg`, `.png`, `.bmp` です.

### 出力

`--output` 配下に以下を出力します.

- `images/<入力ファイル名 stem>.png`: SHAP 可視化. `topk > 1` のときは top-K が同一図内に並ぶ
- `predictions.csv`: 画像ごとの予測結果サマリ
  ```csv
  image_path,predicted_classes,predicted_scores,shap_png
  ./val/cat_001.jpg,"[2]","[0.82]",./out/images/cat_001.png
  ```
  `predicted_classes` / `predicted_scores` は top-K のリストを JSON 文字列で格納する.

### 失敗時挙動

- 入力パス欠落, 値域エラー, config 不正は終了コード `2` で終了する
- それ以外の実行時エラー (モデル読み込み失敗など) は終了コード `1` で終了する

## 実行例

サンプル config として [`configs/example.json`](configs/example.json) を同梱しています.
ImageNet 既定の mean / std と ResNet18 224×224 の典型値を入れた動作する config です.
自プロジェクト向けには, これをコピーして `model` / `background_dir` / 学習時の前処理に合わせて書き換えてください.

```bash
cp configs/example.json ./config.json
# config.json を編集後
uv run pochi \
  --config ./config.json \
  --data ./val_subset \
  --output ./out
```

CLI で一部だけ上書きしたい場合:

```bash
uv run pochi \
  --config ./config.json \
  --data ./val_subset \
  --output ./out \
  --topk 3 \
  --nsamples 50 \
  --device cuda
```
