"""画像分類向け SHAP 実行パイプライン."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExplainRequest", "explain_dataset"]

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ExplainRequest:
    """1 回分の explain 実行パラメータ."""

    model_path: Path
    data_dir: Path
    output_dir: Path
    topk: int
    nsamples: int


def explain_dataset(request: ExplainRequest) -> Path:
    """データセットに対して SHAP パイプラインを実行する.

    Args:
        request: 実行パラメータ.

    Returns:
        出力した `predictions.csv` のパス.

    Raises:
        FileNotFoundError: モデルファイルまたは入力ディレクトリが存在しない場合.
        ValueError: 引数値が不正, または対応画像が存在しない場合.
        NotImplementedError: モデル読み込み実装が未完了の場合.
    """
    _validate_request(request)
    images = _collect_images(request.data_dir)
    output_images_dir = request.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = request.output_dir / "predictions.csv"

    # Model loading and SHAP computation will be implemented in the next step.
    _ = _build_model_predictor(request.model_path)

    with predictions_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "predicted_class", "score", "shap_png"])
        for image_path in images:
            writer.writerow([str(image_path), "", "", ""])

    return predictions_csv_path


def _validate_request(request: ExplainRequest) -> None:
    """実行引数の妥当性を検証する.

    Args:
        request: 実行パラメータ.

    Raises:
        FileNotFoundError: モデルファイルまたは入力ディレクトリが存在しない場合.
        ValueError: `topk` または `nsamples` が 1 未満の場合.
    """
    if not request.model_path.is_file():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {request.model_path}")
    if not request.data_dir.is_dir():
        raise FileNotFoundError(
            f"入力データディレクトリが見つかりません: {request.data_dir}"
        )
    if request.topk < 1:
        raise ValueError("--topk は 1 以上を指定してください")
    if request.nsamples < 1:
        raise ValueError("--nsamples は 1 以上を指定してください")
    request.output_dir.mkdir(parents=True, exist_ok=True)


def _collect_images(data_dir: Path) -> list[Path]:
    """入力ディレクトリ配下の対応画像を再帰収集する.

    Args:
        data_dir: 入力画像ディレクトリ.

    Returns:
        対応画像パスの昇順リスト.

    Raises:
        ValueError: 対応画像が 1 枚も見つからない場合.
    """
    images = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"対応画像が見つかりませんでした: {data_dir}")
    return images


def _build_model_predictor(model_path: Path) -> object:
    """モデルファイルから推論器を構築する.

    Args:
        model_path: モデルファイルのパス.

    Raises:
        NotImplementedError: 実装が未完了の場合.
    """
    raise NotImplementedError(
        "モデル読み込みと SHAP 実行は未実装です. " f"対象モデル: {model_path}"
    )
