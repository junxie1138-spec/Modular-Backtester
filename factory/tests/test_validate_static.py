from pathlib import Path

import pytest

from factory.validate import StaticValidationError, validate_static

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_valid_strategy_passes() -> None:
    validate_static(
        strategy_id="gen_test_valid",
        strategy_src=_read("valid_strategy.py"),
        config_src=_read("valid_config.yaml"),
        allow_short=False,
    )


def test_missing_shift_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_no_shift.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "shift(1)" in str(exc.value)


def test_forbidden_imports_fail() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_bad_imports.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    msg = str(exc.value).lower()
    assert "import" in msg and ("os" in msg or "requests" in msg)


def test_missing_class_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_no_class.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "GeneratedStrategy" in str(exc.value)


def test_config_strategy_mismatch_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("valid_strategy.py"),
            config_src=_read("invalid_config_wrong_strategy.yaml"),
            allow_short=False,
        )
    msg = str(exc.value).lower()
    assert "strategy" in msg


def test_strategy_id_attribute_must_match_injected_id() -> None:
    # valid_strategy.py declares strategy_id = "gen_test_valid"; passing a
    # different injected id is a mismatch.
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_different",
            strategy_src=_read("valid_strategy.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "strategy_id" in str(exc.value)


def test_multi_symbol_attribute_is_forbidden() -> None:
    poisoned = _read("valid_strategy.py").replace(
        'strategy_id = "gen_test_valid"',
        'strategy_id = "gen_test_valid"\n    uses_multi_symbol = True',
    )
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=poisoned,
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "uses_multi_symbol" in str(exc.value)
