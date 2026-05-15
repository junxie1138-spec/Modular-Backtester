from pathlib import Path

import pytest

from factory.filesystem import (
    FilesystemError,
    pick_unused_strategy_id,
    write_strategy_artifacts,
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
