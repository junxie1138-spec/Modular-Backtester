import logging
import random
from pathlib import Path
from unittest import mock

import pytest

from factory.loop import _model_from_flags, configure_logging, run_loop
from factory.settings_loader import load_settings


def test_model_from_flags_extracts_separate_arg() -> None:
    flags = ("-p", "--model", "sonnet", "--allowedTools", "Read")
    assert _model_from_flags(flags) == "sonnet"


def test_model_from_flags_extracts_equals_form() -> None:
    assert _model_from_flags(("-p", "--model=haiku")) == "haiku"


def test_model_from_flags_defaults_when_absent() -> None:
    flags = ("-p", "--output-format", "json", "--allowedTools", "Read")
    assert _model_from_flags(flags) == "(Claude Code default)"


def test_configure_logging_creates_rotating_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "factory.log"
    configure_logging(log_path)
    root = logging.getLogger("factory")
    assert any(
        "RotatingFileHandler" in type(h).__name__ for h in root.handlers
    )


def test_run_loop_runs_max_cycles_then_exits(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    s = load_settings(tmp_settings_file)
    assert s.loop.max_cycles == 1   # from the test fixture

    from factory.cycle import CycleOutcome
    fake_outcome = CycleOutcome(status="failed", failed_stage="generation",
                                strategy_id=None, record={"status": "failed"})
    with mock.patch("factory.loop.run_cycle", return_value=fake_outcome) as rc:
        completed = run_loop(s, rng=random.Random(0))
    assert rc.call_count == 1
    assert completed == 1


def test_run_loop_stops_on_sigint(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    # Override max_cycles to 0 (unbounded) and inject SIGINT after first cycle.
    from factory.loop import _ShutdownFlag, run_loop
    from factory.cycle import CycleOutcome

    flag = _ShutdownFlag()
    outcome = CycleOutcome(status="complete", failed_stage=None,
                           strategy_id="gen_x", record={"status": "complete"})

    call_count = {"n": 0}
    def fake_cycle(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            flag.set()
        return outcome

    with mock.patch("factory.loop.run_cycle", side_effect=fake_cycle):
        completed = run_loop(
            s, rng=random.Random(0), shutdown_flag=flag, max_cycles_override=0,
        )
    # The flag is checked AFTER each cycle, so cycle 2 runs to completion
    # and then the loop breaks.
    assert completed == 2


def test_run_loop_runs_sortino_migration_and_drain(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    assert s.loop.max_cycles == 1   # from the test fixture

    from factory.cycle import CycleOutcome
    fake_outcome = CycleOutcome(status="failed", failed_stage="generation",
                                strategy_id=None, record={"status": "failed"})
    with mock.patch("factory.loop.run_cycle", return_value=fake_outcome), \
         mock.patch("factory.loop.migrate_shard") as migrate, \
         mock.patch("factory.loop.drain_one_retro_promotion") as drain:
        run_loop(s, rng=random.Random(0))

    assert migrate.call_count == 1   # once, at startup
    assert drain.call_count == 1     # once per cycle (max_cycles=1)
