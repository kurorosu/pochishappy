"""pochishappy のロガー設定."""

from __future__ import annotations

import logging

__all__ = ["get_logger"]

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s|%(levelname)-5.5s|%(module)-18s|%(lineno)03d| %(message)s"


def get_logger(name: str) -> logging.Logger:
    """指定名のロガーを取得する.

    Args:
        name: ロガー名.

    Returns:
        フォーマット設定済みロガー.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
