"""CLI behavior tests."""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet18

from pochishappy import cli


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    """テスト用 config.json を書き出すヘルパー."""
    payload: dict[str, object] = {
        "model": "./model.pth",
        "arch": "resnet18",
        "background_dir": "./background",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 32,
        "topk": 1,
        "nsamples": 1,
    }
    payload.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _write_dummy_image(path: Path) -> None:
    """単色 RGB ダミー画像を `path` に書き出す."""
    Image.new("RGB", (64, 64), (128, 128, 128)).save(path)


def _save_dummy_resnet18(path: Path, num_classes: int = 2) -> None:
    """テスト用 resnet18 state_dict を `path` に保存する."""
    model = resnet18(weights=None, num_classes=num_classes)
    torch.save(model.state_dict(), path)


def test_main_missing_model_returns_exit_code_2(tmp_path: Path) -> None:
    """モデルファイルが存在しない場合に exit 2."""
    config_path = _write_config(tmp_path)
    background_dir = tmp_path / "background"
    background_dir.mkdir()
    _write_dummy_image(background_dir / "bg.png")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "sample.png")

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2


def test_main_missing_background_returns_exit_code_2(tmp_path: Path) -> None:
    """background ディレクトリが存在しない場合に exit 2."""
    config_path = _write_config(tmp_path)
    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "sample.png")

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2


def test_main_without_images_returns_exit_code_2(tmp_path: Path) -> None:
    """入力ディレクトリに画像が無い場合に exit 2."""
    config_path = _write_config(tmp_path)
    model_path = tmp_path / "model.pth"
    _save_dummy_resnet18(model_path)
    background_dir = tmp_path / "background"
    background_dir.mkdir()
    _write_dummy_image(background_dir / "bg.png")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2


def test_main_invalid_model_file_returns_exit_code_1(tmp_path: Path) -> None:
    """不正な .pth (state_dict として読み込めない) で exit 1 が返る."""
    config_path = _write_config(tmp_path)
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"not-a-valid-pth-file")
    background_dir = tmp_path / "background"
    background_dir.mkdir()
    _write_dummy_image(background_dir / "bg.png")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "sample.png")

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_main_missing_config_returns_exit_code_2(tmp_path: Path) -> None:
    """config ファイルが存在しない場合に exit 2."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dummy_image(data_dir / "sample.png")

    exit_code = cli.main(
        [
            "--config",
            str(tmp_path / "missing.json"),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 2


def test_main_cli_topk_overrides_config(tmp_path: Path) -> None:
    """CLI の --topk が config 値を上書きする (_build_request の解決ロジック単体)."""
    config_path = _write_config(tmp_path, topk=1)
    config = cli.load_config(config_path)

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--data",
            ".",
            "--output",
            ".",
            "--topk",
            "5",
        ]
    )

    request = cli._build_request(args, config)

    assert request.topk == 5


def test_main_cli_topk_unset_uses_config(tmp_path: Path) -> None:
    """CLI の --topk 未指定なら config 値を使う."""
    config_path = _write_config(tmp_path, topk=3)
    config = cli.load_config(config_path)

    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--data",
            ".",
            "--output",
            ".",
        ]
    )

    request = cli._build_request(args, config)

    assert request.topk == 3
