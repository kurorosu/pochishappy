# pochishappy
Make your SHAP analysis happy — Pochi explains it all with a wagging tail!

## MVP

`pochishappy` は画像分類モデル向けの SHAP 可視化を生成する CLI ツールです.

## 入出力契約

### 入力

- `--model`: `.pth` モデルファイルのパス.
- `--data`: 入力画像ディレクトリ. サブディレクトリを再帰探索します.
- `--output`: 出力先ディレクトリ.
- `--topk`: 説明対象にする上位予測クラス数.
- `--nsamples`: SHAP サンプル数.

対応画像拡張子は `.jpg`, `.jpeg`, `.png`, `.bmp` です.

### 出力

`--output` には以下を出力します.

- `images/`: SHAP 可視化 PNG (`*.png`).
- `predictions.csv`: 画像ごとの予測結果サマリ.

### 失敗時挙動

- 入力パス不足や不正値は, エラーメッセージを表示して終了コード `2` で終了します.
- 実行時エラーは, エラーメッセージを表示して終了コード `1` で終了します.

## 実行例

```bash
uv run pochi \
  --model ./model.pth \
  --data ./sample_images \
  --output ./out \
  --topk 1 \
  --nsamples 20
```
