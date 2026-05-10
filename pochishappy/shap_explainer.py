"""画像分類向け SHAP 実行パイプライン."""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shap
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from pochishappy.config import SUPPORTED_ARCHS
from pochishappy.logger import get_logger

__all__ = ["ExplainRequest", "PredictionRow", "explain_dataset"]

_LOGGER = get_logger("pochishappy.shap_explainer")

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
_MAX_BACKGROUND_IMAGES: int = 50


@dataclass(frozen=True)
class ExplainRequest:
    """1 回分の explain 実行パラメータ."""

    model_path: Path
    arch: str
    data_dir: Path
    background_dir: Path
    output_dir: Path
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_size: int
    resize: int | None
    topk: int
    nsamples: int
    device: torch.device


@dataclass(frozen=True)
class PredictionRow:
    """predictions.csv 1 行分のデータ."""

    image_path: Path
    predicted_classes: list[int]
    predicted_scores: list[float]
    shap_png: Path


@dataclass(frozen=True)
class _Predictor:
    """モデルと前処理をひとまとめにした内部表現."""

    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    num_classes: int


def explain_dataset(request: ExplainRequest) -> Path:
    """データセットに対して SHAP パイプラインを実行する.

    Args:
        request: 実行パラメータ.

    Returns:
        出力した `predictions.csv` のパス.

    Raises:
        FileNotFoundError: モデルファイル, 入力ディレクトリ, background ディレクトリのいずれかが存在しない場合.
        ValueError: 引数値が不正, 対応画像が存在しない, または未サポート arch.
    """
    _validate_request(request)
    images = _collect_images(request.data_dir)
    background_images = _collect_background_images(
        request.background_dir, max_count=_MAX_BACKGROUND_IMAGES
    )

    output_images_dir = request.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv_path = request.output_dir / "predictions.csv"

    predictor = _build_predictor(request)
    background_tensor = _build_background_tensor(predictor, background_images)
    explainer = _build_explainer(predictor.model, background_tensor)

    rows: list[PredictionRow] = []
    for image_path in images:
        row = _explain_one(
            predictor=predictor,
            explainer=explainer,
            image_path=image_path,
            topk=request.topk,
            nsamples=request.nsamples,
            output_dir=request.output_dir,
        )
        rows.append(row)
        _LOGGER.info(
            "processed: %s -> top1=%d (%.3f)",
            image_path.name,
            row.predicted_classes[0],
            row.predicted_scores[0],
        )

    _write_predictions_csv(rows, predictions_csv_path)
    return predictions_csv_path


def _validate_request(request: ExplainRequest) -> None:
    """実行引数の妥当性を検証する.

    Args:
        request: 実行パラメータ.

    Raises:
        FileNotFoundError: 必須パスが存在しない場合.
        ValueError: 値域不正, または未サポート arch.
    """
    if not request.model_path.is_file():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {request.model_path}")
    if not request.data_dir.is_dir():
        raise FileNotFoundError(
            f"入力データディレクトリが見つかりません: {request.data_dir}"
        )
    if not request.background_dir.is_dir():
        raise FileNotFoundError(
            f"background ディレクトリが見つかりません: {request.background_dir}"
        )
    if request.topk < 1:
        raise ValueError("topk は 1 以上を指定してください")
    if request.nsamples < 1:
        raise ValueError("nsamples は 1 以上を指定してください")
    if request.arch not in SUPPORTED_ARCHS:
        raise ValueError(f"未サポートの arch: {request.arch!r}")
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


def _collect_background_images(background_dir: Path, max_count: int) -> list[Path]:
    """Background ディレクトリから先頭 `max_count` 枚を取得する.

    Args:
        background_dir: background 画像ディレクトリ.
        max_count: 最大読み込み枚数.

    Returns:
        対応画像パスのリスト (最大 max_count 件).

    Raises:
        ValueError: 対応画像が 1 枚も見つからない場合.
    """
    images = sorted(
        path
        for path in background_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"background 画像が見つかりませんでした: {background_dir}")
    return images[:max_count]


def _build_transform(
    input_size: int,
    resize: int | None,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Callable[[Image.Image], torch.Tensor]:
    """Torchvision の前処理パイプラインを構築する.

    `resize` 指定時は `Resize(resize) → CenterCrop(input_size)`, 省略時は
    `Resize((input_size, input_size))` で直接リサイズする.

    Args:
        input_size: モデル入力の最終サイズ.
        resize: 中間リサイズサイズ. `None` の場合は直接リサイズ.
        mean: 正規化 mean.
        std: 正規化 std.

    Returns:
        画像を `torch.Tensor` に変換する transform.
    """
    if resize is not None:
        return transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _load_state_dict_with_num_classes(
    model_path: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    """`.pth` から state_dict を読み込み, `fc.out_features` から num_classes を推定する.

    Args:
        model_path: `.pth` モデルファイルのパス.

    Returns:
        `(state_dict, num_classes)` のタプル.

    Raises:
        ValueError: state_dict に `fc.weight` が含まれていない場合.
    """
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "fc.weight" not in state_dict:
        raise ValueError(
            "state_dict に fc.weight が含まれていません. "
            f"ResNet 系の重みを指定してください: {model_path}"
        )
    num_classes = int(state_dict["fc.weight"].shape[0])
    return state_dict, num_classes


def _build_predictor(request: ExplainRequest) -> _Predictor:
    """`ExplainRequest` からモデルと前処理をひとまとめにした `_Predictor` を構築する.

    Args:
        request: 実行パラメータ.

    Returns:
        `_Predictor`.

    Raises:
        ValueError: 未サポートの arch, または state_dict 不整合.
    """
    if request.arch not in SUPPORTED_ARCHS:
        raise ValueError(f"未サポートの arch: {request.arch!r}")

    state_dict, num_classes = _load_state_dict_with_num_classes(request.model_path)
    model = resnet18(weights=None, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(request.device)

    transform = _build_transform(
        input_size=request.input_size,
        resize=request.resize,
        mean=request.mean,
        std=request.std,
    )
    return _Predictor(
        model=model,
        transform=transform,
        device=request.device,
        num_classes=num_classes,
    )


def _build_background_tensor(
    predictor: _Predictor, background_images: list[Path]
) -> torch.Tensor:
    """Background 画像群を transform 適用してテンソルにまとめる.

    Args:
        predictor: transform / device を保持する `_Predictor`.
        background_images: 読み込む画像パスのリスト.

    Returns:
        shape `(N, C, H, W)` のテンソル.
    """
    tensors: list[torch.Tensor] = []
    for path in background_images:
        image = Image.open(path).convert("RGB")
        tensors.append(predictor.transform(image))
    return torch.stack(tensors).to(predictor.device)


def _build_explainer(
    model: nn.Module, background_tensor: torch.Tensor
) -> shap.GradientExplainer:
    """`GradientExplainer` を構築する.

    Args:
        model: 学習済みモデル.
        background_tensor: reference 分布となる入力テンソル.

    Returns:
        `shap.GradientExplainer`.
    """
    return shap.GradientExplainer(model, background_tensor)


def _explain_one(
    predictor: _Predictor,
    explainer: shap.GradientExplainer,
    image_path: Path,
    topk: int,
    nsamples: int,
    output_dir: Path,
) -> PredictionRow:
    """1 枚の画像に対して SHAP 計算と PNG 保存を行う.

    Args:
        predictor: モデルと前処理.
        explainer: `GradientExplainer`.
        image_path: 入力画像のパス.
        topk: 上位予測クラス数.
        nsamples: SHAP の推定試行回数.
        output_dir: 出力ディレクトリ. 配下に `images/<stem>.png` を保存する.

    Returns:
        `PredictionRow`.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)

    effective_topk = min(topk, predictor.num_classes)

    with torch.no_grad():
        logits = predictor.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_score, topk_idx = probs.topk(effective_topk)

    shap_result: Any = explainer.shap_values(
        tensor, ranked_outputs=effective_topk, nsamples=nsamples
    )
    # ranked_outputs 指定時は (shap_values, top_idx) のタプルが返る.
    # shap_values shape: (N, C, H, W, K)
    if isinstance(shap_result, tuple):
        shap_values, _ = shap_result
    else:
        shap_values = shap_result

    # shap.image_plot は channels-last (NHWC) 形式を期待するので転置する.
    shap_values_hwc = shap_values.transpose(0, 2, 3, 1, 4)
    pixel_values_hwc = tensor.cpu().numpy().transpose(0, 2, 3, 1)

    # plt.gcf() に描画する. show=False で画面表示を抑制し savefig で取り出す.
    plt.figure()
    shap.image_plot(shap_values_hwc, pixel_values_hwc, show=False)
    png_path = output_dir / "images" / f"{image_path.stem}.png"
    plt.gcf().savefig(png_path, bbox_inches="tight")
    plt.close()

    return PredictionRow(
        image_path=image_path,
        predicted_classes=[int(idx) for idx in topk_idx.tolist()],
        predicted_scores=[float(score) for score in topk_score.tolist()],
        shap_png=png_path,
    )


def _write_predictions_csv(rows: list[PredictionRow], csv_path: Path) -> None:
    """`PredictionRow` のリストを CSV に書き出す.

    `predicted_classes` / `predicted_scores` は JSON 文字列で 1 セルに格納する.

    Args:
        rows: 書き出す行のリスト.
        csv_path: 出力先 CSV パス.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["image_path", "predicted_classes", "predicted_scores", "shap_png"]
        )
        for row in rows:
            writer.writerow(
                [
                    str(row.image_path),
                    json.dumps(row.predicted_classes),
                    json.dumps(row.predicted_scores),
                    str(row.shap_png),
                ]
            )
