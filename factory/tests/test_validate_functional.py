from pathlib import Path

import pytest

from factory.validate import FunctionalValidationError, validate_functional

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.mark.slow
def test_valid_strategy_passes_functional(tmp_path: Path) -> None:
    validate_functional(
        strategy_id="gen_test_valid",
        strategy_src=_read("valid_strategy.py"),
        allow_short=False,
        tmp_dir=tmp_path,
    )


@pytest.mark.slow
def test_bad_signal_dtype_fails(tmp_path: Path) -> None:
    with pytest.raises(FunctionalValidationError) as exc:
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_signal_dtype.py"),
            allow_short=False,
            tmp_dir=tmp_path,
        )
    msg = str(exc.value).lower()
    assert "signal" in msg and ("int" in msg or "dtype" in msg)


@pytest.mark.slow
def test_short_signal_under_long_only_fails(tmp_path: Path) -> None:
    with pytest.raises(FunctionalValidationError) as exc:
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_signal_short.py"),
            allow_short=False,
            tmp_dir=tmp_path,
        )
    msg = str(exc.value).lower()
    assert "-1" in msg or "short" in msg or "long-only" in msg


@pytest.mark.slow
def test_unimportable_strategy_fails(tmp_path: Path) -> None:
    # Trip an import-time error: bad syntax inside the body.
    bad = "from __future__ import annotations\nthis is not valid python\n"
    with pytest.raises(FunctionalValidationError):
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=bad,
            allow_short=False,
            tmp_dir=tmp_path,
        )
