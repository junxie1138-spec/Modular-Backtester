from pathlib import Path

import pytest

from factory.filesystem import (
    FilesystemError,
    RegistryAlreadyHasStrategy,
    append_registry_entry,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)


def _seed_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from __future__ import annotations\n"
        "from backtester.strategies.base import BaseStrategy\n"
        "STRATEGY_REGISTRY = {}\n"
        "def register_strategy(cls): STRATEGY_REGISTRY[cls.strategy_id] = cls; return cls\n",
        encoding="utf-8",
    )


def test_write_strategy_and_config(tmp_path: Path) -> None:
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs" / "wfo"
    write_strategy_artifacts(
        strategy_id="gen_42",
        strategy_src="# strategy body\n",
        config_src="run_name: gen_42\n",
        strategies_dir=strat_dir,
        configs_dir=cfg_dir,
    )
    assert (strat_dir / "gen_42.py").read_text(encoding="utf-8") == "# strategy body\n"
    assert (cfg_dir / "gen_42.yaml").read_text(encoding="utf-8") == "run_name: gen_42\n"


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "gen_42.py").write_text("existing", encoding="utf-8")
    with pytest.raises(FilesystemError) as exc:
        write_strategy_artifacts(
            strategy_id="gen_42",
            strategy_src="# new",
            config_src="run_name: gen_42\n",
            strategies_dir=tmp_path / "strategies",
            configs_dir=tmp_path / "configs" / "wfo",
        )
    assert "exists" in str(exc.value).lower()


def test_append_registry_entry_adds_two_lines(tmp_path: Path) -> None:
    reg = tmp_path / "backtester" / "strategies" / "registry.py"
    _seed_registry(reg)
    append_registry_entry(strategy_id="gen_42", registry_file=reg)
    text = reg.read_text(encoding="utf-8")
    assert "from strategies.gen_42 import GeneratedStrategy as _gen_42" in text
    assert "register_strategy(_gen_42)" in text


def test_append_registry_is_idempotent(tmp_path: Path) -> None:
    reg = tmp_path / "backtester" / "strategies" / "registry.py"
    _seed_registry(reg)
    append_registry_entry(strategy_id="gen_42", registry_file=reg)
    with pytest.raises(RegistryAlreadyHasStrategy):
        append_registry_entry(strategy_id="gen_42", registry_file=reg)


def test_append_registry_does_not_false_positive_on_prefix_match(tmp_path: Path) -> None:
    """Regression: gen_42's alias _gen_42 must not be detected as already-present
    when only the longer-suffix _gen_42_2 is registered.
    """
    reg = tmp_path / "backtester" / "strategies" / "registry.py"
    _seed_registry(reg)
    # Register the longer-suffixed strategy first.
    append_registry_entry(strategy_id="gen_42_2", registry_file=reg)
    # Now register the shorter base id; this must succeed.
    append_registry_entry(strategy_id="gen_42", registry_file=reg)
    text = reg.read_text(encoding="utf-8")
    # Both aliases must be present.
    assert "register_strategy(_gen_42_2)" in text
    assert "register_strategy(_gen_42)" in text


def test_pick_unused_strategy_id_returns_base_when_free(tmp_path: Path) -> None:
    strat = tmp_path / "strategies"
    strat.mkdir()
    assert pick_unused_strategy_id("gen_42", strategies_dir=strat) == "gen_42"


def test_pick_unused_strategy_id_bumps_on_collision(tmp_path: Path) -> None:
    strat = tmp_path / "strategies"
    strat.mkdir()
    (strat / "gen_42.py").write_text("x", encoding="utf-8")
    (strat / "gen_42_2.py").write_text("x", encoding="utf-8")
    assert pick_unused_strategy_id("gen_42", strategies_dir=strat) == "gen_42_3"
