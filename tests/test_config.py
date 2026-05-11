"""JSON config 読み込みのテスト."""

import json
from pathlib import Path

import pytest

from pochishappy.config import PochiConfig, load_config


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    """テスト用の config.json を書き出すヘルパー.

    Args:
        tmp_path: pytest tmp_path フィクスチャ.
        **overrides: 既定値を上書きしたいキーと値.

    Returns:
        書き出した config ファイルのパス.
    """
    payload: dict[str, object] = {
        "model": "./model.pth",
        "arch": "resnet18",
        "background_dir": "./background",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224,
        "resize": 256,
        "topk": 1,
        "nsamples": 20,
    }
    payload.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def test_load_config_returns_pochi_config(tmp_path: Path) -> None:
    """正常な JSON が PochiConfig にマップされる."""
    config_path = _write_config(tmp_path)

    config = load_config(config_path)

    assert isinstance(config, PochiConfig)
    assert config.arch == "resnet18"
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)
    assert config.input_size == 224
    assert config.resize == 256
    assert config.topk == 1
    assert config.nsamples == 20


def test_load_config_resize_omitted_becomes_none(tmp_path: Path) -> None:
    """resize 省略時は None になる."""
    payload = {
        "model": "./model.pth",
        "arch": "resnet18",
        "background_dir": "./background",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "input_size": 224,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.resize is None
    assert config.topk == 1
    assert config.nsamples == 20


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    """相対パスは config ファイルのディレクトリ基準で解決される."""
    config_path = _write_config(tmp_path)

    config = load_config(config_path)

    assert config.model_path == (tmp_path / "model.pth").resolve()
    assert config.background_dir == (tmp_path / "background").resolve()


def test_load_config_keeps_absolute_paths(tmp_path: Path) -> None:
    """絶対パスはそのまま保持される."""
    absolute_model = tmp_path / "absolute_dir" / "model.pth"
    config_path = _write_config(tmp_path, model=str(absolute_model))

    config = load_config(config_path)

    assert config.model_path == absolute_model


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """存在しない config パスで FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "missing_key",
    ["model", "arch", "background_dir", "mean", "std", "input_size"],
)
def test_load_config_missing_required_key_raises(
    tmp_path: Path, missing_key: str
) -> None:
    """必須キー欠落で ValueError."""
    payload: dict[str, object] = {
        "model": "./model.pth",
        "arch": "resnet18",
        "background_dir": "./background",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "input_size": 224,
    }
    payload.pop(missing_key)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="必須キー"):
        load_config(config_path)


def test_load_config_unsupported_arch_raises(tmp_path: Path) -> None:
    """未サポートの arch で ValueError."""
    config_path = _write_config(tmp_path, arch="vit_b_16")

    with pytest.raises(ValueError, match="未サポートの arch"):
        load_config(config_path)


@pytest.mark.parametrize("bad_mean", [[0.5, 0.5], [0.5, 0.5, 0.5, 0.5], "not a list"])
def test_load_config_invalid_mean_length_raises(
    tmp_path: Path, bad_mean: object
) -> None:
    """mean の長さが 3 でないとき ValueError."""
    config_path = _write_config(tmp_path, mean=bad_mean)

    with pytest.raises(ValueError, match="mean"):
        load_config(config_path)


def test_load_config_non_numeric_mean_raises(tmp_path: Path) -> None:
    """mean に数値以外が含まれるとき ValueError."""
    config_path = _write_config(tmp_path, mean=[0.5, "x", 0.5])

    with pytest.raises(ValueError, match="数値"):
        load_config(config_path)


@pytest.mark.parametrize("key", ["input_size", "topk", "nsamples"])
def test_load_config_non_positive_int_raises(tmp_path: Path, key: str) -> None:
    """input_size / topk / nsamples が 0 以下で ValueError."""
    config_path = _write_config(tmp_path, **{key: 0})

    with pytest.raises(ValueError, match=key):
        load_config(config_path)


def test_load_config_resize_smaller_than_input_size_raises(tmp_path: Path) -> None:
    """resize < input_size で ValueError."""
    config_path = _write_config(tmp_path, input_size=224, resize=200)

    with pytest.raises(ValueError, match="resize"):
        load_config(config_path)


def test_load_config_bool_for_int_raises(tmp_path: Path) -> None:
    """int キーに bool を渡すと ValueError (Python 上 bool は int だが意図しない値)."""
    config_path = _write_config(tmp_path, topk=True)

    with pytest.raises(ValueError, match="topk"):
        load_config(config_path)
