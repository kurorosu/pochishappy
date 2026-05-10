"""pochishappy の CLI エントリポイント."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from pochishappy.config import PochiConfig, load_config
from pochishappy.logger import get_logger
from pochishappy.shap_explainer import ExplainRequest, explain_dataset

__all__ = ["main"]

_LOGGER = get_logger("pochishappy.cli")

_DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "cuda")


def _build_parser() -> ArgumentParser:
    """CLI パーサーを構築する.

    Returns:
        引数定義済みの `ArgumentParser`.
    """
    parser = ArgumentParser(prog="pochi", description="SHAP 可視化を実行")
    parser.add_argument("--config", required=True, help="JSON config ファイルのパス")
    parser.add_argument("--data", required=True, help="入力画像ディレクトリのパス")
    parser.add_argument("--output", required=True, help="出力ディレクトリのパス")
    parser.add_argument(
        "--model", default=None, help="config の model を上書きする .pth パス"
    )
    parser.add_argument(
        "--background-dir",
        dest="background_dir",
        default=None,
        help="config の background_dir を上書きする画像ディレクトリ",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="config の topk を上書きする上位予測クラス数",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="config の nsamples を上書きする SHAP 推定試行回数",
    )
    parser.add_argument(
        "--device",
        choices=_DEVICE_CHOICES,
        default="auto",
        help="推論デバイス. auto は cuda 利用可能なら cuda, 不可なら cpu",
    )
    return parser


def _resolve_device(name: str) -> torch.device:
    """`--device` 引数を `torch.device` に解決する.

    Args:
        name: `"auto"`, `"cpu"`, `"cuda"` のいずれか.

    Returns:
        解決済み `torch.device`.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_request(args: Namespace, config: PochiConfig) -> ExplainRequest:
    """CLI 引数と config から `ExplainRequest` を組み立てる.

    CLI 引数が `None` でない場合は config 値を上書きする (`CLI > config > 既定値`).

    Args:
        args: パース済み CLI 引数.
        config: 読み込み済み `PochiConfig`.

    Returns:
        組み立て済み `ExplainRequest`.
    """
    model_path = Path(args.model) if args.model is not None else config.model_path
    background_dir = (
        Path(args.background_dir)
        if args.background_dir is not None
        else config.background_dir
    )
    topk = args.topk if args.topk is not None else config.topk
    nsamples = args.nsamples if args.nsamples is not None else config.nsamples
    device = _resolve_device(args.device)

    return ExplainRequest(
        model_path=model_path,
        arch=config.arch,
        data_dir=Path(args.data),
        background_dir=background_dir,
        output_dir=Path(args.output),
        mean=config.mean,
        std=config.std,
        input_size=config.input_size,
        resize=config.resize,
        topk=topk,
        nsamples=nsamples,
        device=device,
    )


def _run(args: Namespace) -> int:
    """CLI 実行本体.

    Args:
        args: パース済み引数.

    Returns:
        正常終了時の終了コード `0`.
    """
    config = load_config(Path(args.config))
    request = _build_request(args, config)
    predictions_path = explain_dataset(request)
    _LOGGER.info("生成完了: %s", predictions_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 本体を実行する.

    Args:
        argv: コマンドライン引数. `None` の場合は `sys.argv` を使用.

    Returns:
        実行結果の終了コード.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except (FileNotFoundError, ValueError) as error:
        _LOGGER.error("入力エラー: %s", error)
        return 2
    except Exception as error:  # pragma: no cover - defensive fallback
        _LOGGER.exception("予期しないエラー: %s", error)
        return 1
