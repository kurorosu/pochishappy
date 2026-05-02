"""Package version tests."""

import pochishappy


def test_version() -> None:
    """__version__ が定義されていることを確認する."""
    assert pochishappy.__version__ == "0.1.0"
