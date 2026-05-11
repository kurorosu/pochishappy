"""SHAP explainer のテスト."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import resnet18

from pochishappy.shap_explainer import (
    ExplainRequest,
    _build_predictor,
    _build_transform,
    _collect_background_images,
    _load_state_dict_with_num_classes,
    explain_dataset,
)


def _save_dummy_resnet18(path: Path, num_classes: int = 2) -> None:
    """テスト用に resnet18 state_dict を `path` に保存する."""
    model = resnet18(weights=None, num_classes=num_classes)
    torch.save(model.state_dict(), path)


def _write_dummy_image(
    path: Path, size: int = 64, color: tuple[int, int, int] = (128, 128, 128)
) -> None:
    """単色の RGB ダミー画像を `path` に書き出す."""
    Image.new("RGB", (size, size), color).save(path)


def _build_request(
    *,
    tmp_path: Path,
    model_path: Path,
    data_dir: Path,
    background_dir: Path,
    output_dir: Path,
    arch: str = "resnet18",
    input_size: int = 32,
    resize: int | None = None,
    topk: int = 1,
    nsamples: int = 1,
) -> ExplainRequest:
    """テスト用 ExplainRequest 構築ヘルパー."""
    return ExplainRequest(
        model_path=model_path,
        arch=arch,
        data_dir=data_dir,
        background_dir=background_dir,
        output_dir=output_dir,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        input_size=input_size,
        resize=resize,
        topk=topk,
        nsamples=nsamples,
        device=torch.device("cpu"),
    )


def test_load_state_dict_extracts_num_classes(tmp_path: Path) -> None:
    """state_dict の fc.out_features から num_classes を推定できる."""
    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path, num_classes=5)

    _, num_classes = _load_state_dict_with_num_classes(model_path)

    assert num_classes == 5


def test_load_state_dict_without_fc_raises(tmp_path: Path) -> None:
    """fc.weight が無い state_dict は ValueError."""
    model_path = tmp_path / "model.pth"
    torch.save({"layer1.weight": torch.zeros(3, 3)}, model_path)

    with pytest.raises(ValueError, match="fc.weight"):
        _load_state_dict_with_num_classes(model_path)


def test_build_transform_with_resize_produces_center_crop_shape() -> None:
    """resize 指定時は Resize -> CenterCrop で input_size に揃う."""
    transform = _build_transform(
        input_size=64, resize=72, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    image = Image.new("RGB", (200, 100), (0, 0, 0))

    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)


def test_build_transform_without_resize_directly_resizes() -> None:
    """resize 省略時は (input_size, input_size) に直接リサイズする."""
    transform = _build_transform(
        input_size=48, resize=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    image = Image.new("RGB", (200, 100), (0, 0, 0))

    tensor = transform(image)

    assert tensor.shape == (3, 48, 48)


def test_build_predictor_loads_model_on_cpu(tmp_path: Path) -> None:
    """_build_predictor が CPU device でモデルを構築する."""
    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path, num_classes=3)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "a.png")
    background_dir = tmp_path / "bg"
    background_dir.mkdir()
    _write_dummy_image(background_dir / "b.png")

    request = _build_request(
        tmp_path=tmp_path,
        model_path=model_path,
        data_dir=data_dir,
        background_dir=background_dir,
        output_dir=tmp_path / "out",
    )

    predictor = _build_predictor(request)

    assert predictor.num_classes == 3
    assert predictor.device == torch.device("cpu")
    assert next(predictor.model.parameters()).device.type == "cpu"


def test_build_predictor_unsupported_arch_raises(tmp_path: Path) -> None:
    """未サポート arch で ValueError."""
    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    background_dir = tmp_path / "bg"
    background_dir.mkdir()

    request = _build_request(
        tmp_path=tmp_path,
        model_path=model_path,
        data_dir=data_dir,
        background_dir=background_dir,
        output_dir=tmp_path / "out",
        arch="vit_b_16",
    )

    with pytest.raises(ValueError, match="未サポートの arch"):
        _build_predictor(request)


def test_collect_background_images_caps_at_max_count(tmp_path: Path) -> None:
    """background 画像は max_count まで切り詰められる."""
    background_dir = tmp_path / "bg"
    background_dir.mkdir()
    for i in range(7):
        _write_dummy_image(background_dir / f"img_{i:02d}.png")

    images = _collect_background_images(background_dir, max_count=3)

    assert len(images) == 3
    assert [path.name for path in images] == ["img_00.png", "img_01.png", "img_02.png"]


def test_collect_background_images_empty_dir_raises(tmp_path: Path) -> None:
    """画像が無い background ディレクトリで ValueError."""
    background_dir = tmp_path / "bg"
    background_dir.mkdir()

    with pytest.raises(ValueError, match="background"):
        _collect_background_images(background_dir, max_count=10)


@pytest.mark.slow
def test_explain_dataset_runs_end_to_end(tmp_path: Path) -> None:
    """ResNet18 + 2 枚の画像で SHAP パイプラインが完走し predictions.csv と PNG を出力する."""
    import csv as csv_module
    import json

    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path, num_classes=2)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "input_a.png", color=(255, 0, 0))
    _write_dummy_image(data_dir / "input_b.png", color=(0, 255, 0))

    background_dir = tmp_path / "bg"
    background_dir.mkdir()
    _write_dummy_image(background_dir / "bg_a.png", color=(0, 0, 255))
    _write_dummy_image(background_dir / "bg_b.png", color=(128, 128, 128))

    output_dir = tmp_path / "out"
    request = _build_request(
        tmp_path=tmp_path,
        model_path=model_path,
        data_dir=data_dir,
        background_dir=background_dir,
        output_dir=output_dir,
        input_size=32,
        topk=1,
        nsamples=1,
    )

    predictions_csv_path = explain_dataset(request)

    assert predictions_csv_path == output_dir / "predictions.csv"
    assert predictions_csv_path.is_file()
    assert (output_dir / "images" / "input_a.png").is_file()
    assert (output_dir / "images" / "input_b.png").is_file()

    with predictions_csv_path.open("r", encoding="utf-8") as csv_file:
        rows = list(csv_module.reader(csv_file))
    assert rows[0] == [
        "image_path",
        "predicted_classes",
        "predicted_scores",
        "shap_png",
    ]
    assert len(rows) == 3  # header + 2 entries

    for row in rows[1:]:
        classes = json.loads(row[1])
        scores = json.loads(row[2])
        assert isinstance(classes, list) and len(classes) == 1
        assert isinstance(scores, list) and len(scores) == 1
        assert 0 <= classes[0] < 2
        assert 0.0 <= scores[0] <= 1.0
