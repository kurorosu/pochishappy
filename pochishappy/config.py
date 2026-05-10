"""JSON config の読み込みとバリデーション."""

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PochiConfig", "SUPPORTED_ARCHS", "load_config"]

SUPPORTED_ARCHS: tuple[str, ...] = ("resnet18",)

_REQUIRED_KEYS: tuple[str, ...] = (
    "model",
    "arch",
    "background_dir",
    "mean",
    "std",
    "input_size",
)


@dataclass(frozen=True)
class PochiConfig:
    """config.json から読み込まれた実行パラメータ."""

    model_path: Path
    arch: str
    background_dir: Path
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input_size: int
    resize: int | None = None
    topk: int = 1
    nsamples: int = 20


def load_config(path: Path) -> PochiConfig:
    """JSON を読み込み, 必須キーの検証と型変換を行う.

    相対パス (`model`, `background_dir`) は config ファイルがあるディレクトリ
    基準で解決する.

    Args:
        path: config JSON ファイルのパス.

    Returns:
        パース済み `PochiConfig`.

    Raises:
        FileNotFoundError: config ファイルが存在しない場合.
        ValueError: 必須キー欠落, 型不正, 値域不正の場合.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config ファイルが見つかりません: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config JSON のルートはオブジェクトである必要があります")

    _check_required_keys(raw)

    arch = _parse_arch(raw["arch"])
    mean = _parse_triplet(raw["mean"], key="mean")
    std = _parse_triplet(raw["std"], key="std")
    input_size = _parse_positive_int(raw["input_size"], key="input_size")
    resize = _parse_optional_resize(raw.get("resize"), input_size=input_size)
    topk = _parse_positive_int(raw.get("topk", 1), key="topk")
    nsamples = _parse_positive_int(raw.get("nsamples", 20), key="nsamples")

    base_dir = path.parent
    model_path = _resolve_path(raw["model"], key="model", base_dir=base_dir)
    background_dir = _resolve_path(
        raw["background_dir"], key="background_dir", base_dir=base_dir
    )

    return PochiConfig(
        model_path=model_path,
        arch=arch,
        background_dir=background_dir,
        mean=mean,
        std=std,
        input_size=input_size,
        resize=resize,
        topk=topk,
        nsamples=nsamples,
    )


def _check_required_keys(raw: dict[str, object]) -> None:
    """必須キーが揃っているか検証する.

    Args:
        raw: JSON から読み込んだ辞書.

    Raises:
        ValueError: 必須キーが欠落している場合.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"config に必須キーが不足しています: {missing}")


def _parse_arch(value: object) -> str:
    """`arch` キーをパースし, サポート対象か検証する.

    Args:
        value: JSON 上の `arch` 値.

    Returns:
        検証済みアーキテクチャ名.

    Raises:
        ValueError: 文字列以外, または未サポートのアーキテクチャ.
    """
    if not isinstance(value, str):
        raise ValueError(f"arch は文字列で指定してください: {value!r}")
    if value not in SUPPORTED_ARCHS:
        raise ValueError(
            f"未サポートの arch: {value!r}. " f"サポート対象: {list(SUPPORTED_ARCHS)}"
        )
    return value


def _parse_triplet(value: object, key: str) -> tuple[float, float, float]:
    """3 要素の float リスト/タプルを検証する.

    Args:
        value: JSON 上の値.
        key: エラーメッセージ用のキー名.

    Returns:
        3 要素 float タプル.

    Raises:
        ValueError: 長さが 3 でない, または数値でない要素を含む.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} は長さ 3 のリストで指定してください: {value!r}")
    floats: list[float] = []
    for index, element in enumerate(value):
        if not isinstance(element, (int, float)) or isinstance(element, bool):
            raise ValueError(f"{key}[{index}] は数値で指定してください: {element!r}")
        floats.append(float(element))
    return (floats[0], floats[1], floats[2])


def _parse_positive_int(value: object, key: str) -> int:
    """1 以上の int を検証する.

    Args:
        value: JSON 上の値.
        key: エラーメッセージ用のキー名.

    Returns:
        検証済み int.

    Raises:
        ValueError: int でない, または 1 未満の場合.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} は int で指定してください: {value!r}")
    if value < 1:
        raise ValueError(f"{key} は 1 以上で指定してください: {value}")
    return value


def _parse_optional_resize(value: object, input_size: int) -> int | None:
    """`resize` キーを検証する. 省略時は `None`.

    Args:
        value: JSON 上の `resize` 値 (省略時は `None`).
        input_size: CenterCrop で使う最終サイズ. resize はこれ以上である必要あり.

    Returns:
        検証済み int, または `None`.

    Raises:
        ValueError: int でない, または `resize < input_size` の場合.
    """
    if value is None:
        return None
    resize = _parse_positive_int(value, key="resize")
    if resize < input_size:
        raise ValueError(
            f"resize ({resize}) は input_size ({input_size}) 以上で指定してください"
        )
    return resize


def _resolve_path(value: object, key: str, base_dir: Path) -> Path:
    """文字列パスを `Path` に変換し, 相対パスは `base_dir` 基準で解決する.

    Args:
        value: JSON 上のパス文字列.
        key: エラーメッセージ用のキー名.
        base_dir: 相対パス解決の基準ディレクトリ (config ファイルのある場所).

    Returns:
        解決済み `Path`.

    Raises:
        ValueError: 文字列でない場合.
    """
    if not isinstance(value, str):
        raise ValueError(f"{key} は文字列パスで指定してください: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
