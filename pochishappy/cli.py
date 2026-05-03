"""pochishappy の CLI エントリポイント."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pochishappy.logger import get_logger
from pochishappy.shap_explainer import ExplainRequest, explain_dataset

__all__ = ["main"]

_LOGGER = get_logger("pochishappy.cli")


def _build_parser() -> ArgumentParser:
    """CLI パーサーを構築する.

    Returns:
        サブコマンド定義済みの `ArgumentParser`.
    """
    parser = ArgumentParser(prog="pochi", description="SHAP 可視化を実行")
    parser.add_argument("--model", required=True, help=".pth モデルファイルのパス")
    parser.add_argument("--data", required=True, help="入力画像ディレクトリのパス")
    parser.add_argument("--output", required=True, help="出力ディレクトリのパス")
    parser.add_argument(
        "--topk", type=int, default=1, help="説明対象にする上位予測クラス数"
    )
    parser.add_argument("--nsamples", type=int, default=20, help="SHAP サンプル数")
    return parser


def _run(args: Namespace) -> int:
    """CLI 実行本体.

    Args:
        args: パース済み引数.

    Returns:
        正常終了時の終了コード `0`.
    """
    request = ExplainRequest(
        model_path=Path(args.model),
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        topk=args.topk,
        nsamples=args.nsamples,
    )
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
    except NotImplementedError as error:
        _LOGGER.error("未実装: %s", error)
        return 1
    except Exception as error:  # pragma: no cover - defensive fallback
        _LOGGER.exception("予期しないエラー: %s", error)
        return 1

    return 0
