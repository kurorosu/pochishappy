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

```bash
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
