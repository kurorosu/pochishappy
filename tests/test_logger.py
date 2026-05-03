"""ロガー設定のテスト."""

from __future__ import annotations

from pochishappy.logger import LOG_FORMAT, get_logger


def test_get_logger_uses_pochitrain_like_format() -> None:
    """pochitrain と同形式のフォーマットを使うことを確認する."""
    logger = get_logger("pochishappy.test")

    assert logger.handlers
    formatter = logger.handlers[0].formatter
    assert formatter is not None
    assert formatter._fmt == LOG_FORMAT
