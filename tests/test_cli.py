"""CLI behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pochishappy import cli


def test_main_explain_invalid_paths(tmp_path: Path) -> None:
    """存在しない入力パス時にエラー終了することを確認する."""
    exit_code = cli.main(
        [
            "--model",
            str(tmp_path / "missing-model.pth"),
            "--data",
            str(tmp_path / "missing-data"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2


def test_main_explain_not_implemented(tmp_path: Path) -> None:
    """未実装部分の例外が適切に伝播されることを確認する."""
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image_path = data_dir / "sample.png"
    image_path.write_bytes(b"fake-image")

    exit_code = cli.main(
        [
            "--model",
            str(model_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


def test_main_explain_without_images(tmp_path: Path) -> None:
    """画像が無い入力ディレクトリでエラー終了することを確認する."""
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    exit_code = cli.main(
        [
            "--model",
            str(model_path),
            "--data",
            str(data_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2
