import logging
import random
import threading
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from factory.loop import _model_from_flags, configure_logging, main, run_loop
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


def test_run_loop_stops_when_generator_needs_login(tmp_settings_file: Path) -> None:
    from factory.generate import GeneratorUnavailable

    s = load_settings(tmp_settings_file)
    with mock.patch("factory.loop.run_cycle", side_effect=GeneratorUnavailable("log in")) as rc, \
         pytest.raises(GeneratorUnavailable, match="log in"):
        run_loop(s, rng=random.Random(0), max_cycles_override=0)
    rc.assert_called_once()


def test_run_loop_stops_when_pool_is_exhausted(tmp_settings_file: Path) -> None:
    from factory.slot_pool import SlotPoolExhausted

    s = load_settings(tmp_settings_file)
    with mock.patch("factory.loop.run_cycle", side_effect=SlotPoolExhausted("empty")) as rc:
        assert run_loop(s, rng=random.Random(0), max_cycles_override=0) == 0
    rc.assert_called_once()


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


def test_main_passes_max_cycles_override(tmp_settings_file: Path) -> None:
    with mock.patch("factory.loop.run_loop") as rl, \
         mock.patch("factory.loop.configure_logging"):
        rc = main(["--settings", str(tmp_settings_file), "--max-cycles", "3"])
    assert rc == 0
    assert rl.call_args.kwargs["max_cycles_override"] == 3


def test_main_returns_failure_when_generator_unavailable(tmp_settings_file: Path) -> None:
    from factory.generate import GeneratorUnavailable

    with mock.patch("factory.loop.run_loop", side_effect=GeneratorUnavailable("log in")), \
         mock.patch("factory.loop.configure_logging"):
        assert main(["--settings", str(tmp_settings_file)]) == 1


def test_main_max_cycles_defaults_to_none(tmp_settings_file: Path) -> None:
    with mock.patch("factory.loop.run_loop") as rl, \
         mock.patch("factory.loop.configure_logging"):
        main(["--settings", str(tmp_settings_file)])
    assert rl.call_args.kwargs["max_cycles_override"] is None


def test_main_rejects_negative_max_cycles(tmp_settings_file: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--settings", str(tmp_settings_file), "--max-cycles", "-1"])
    assert exc.value.code != 0


def test_parallel_workers_share_pool_and_keep_separate_shards(tmp_settings_file: Path) -> None:
    from factory.cycle import CycleOutcome
    from factory.results import read_records, write_record
    from factory.slot_pool import claim_slot_combination, consume_slot_combination

    s = load_settings(tmp_settings_file)
    barrier = threading.Barrier(2, timeout=10)

    def cycle(worker, *, rng):
        claim = claim_slot_combination(worker.paths.slot_pool_db, rng=rng, node_id=worker.node_id)
        # Both workers must have a distinct active claim before either finishes.
        barrier.wait()
        record = {"strategy_id": claim.combination_id, "node_id": worker.node_id,
                  "tmp_dir": str(worker.paths.tmp_dir)}
        write_record(worker.paths.results_dir, record, node_id=worker.node_id)
        consume_slot_combination(worker.paths.slot_pool_db,
                                 combination_id=claim.combination_id,
                                 claimed_by=claim.claimed_by, claimed_at=claim.claimed_at,
                                 strategy_id=claim.combination_id,
                                 outcome_status="complete", failed_stage=None)
        return CycleOutcome("complete", None, claim.combination_id, record)

    with mock.patch("factory.loop.run_cycle", side_effect=cycle), \
         mock.patch("factory.loop._start_discovery_worker", return_value=threading.Event()) as discovery:
        assert run_loop(s, rng=random.Random(0), workers=2, max_cycles_override=4) == 4
    records = read_records(s.paths.results_dir)
    assert len(records) == len({r["strategy_id"] for r in records}) == 4
    assert {r["node_id"] for r in records} == {"local", "local-w2"}
    assert len({r["tmp_dir"] for r in records}) == 2
    discovery.assert_called_once()


@pytest.mark.parametrize("workers,limit", [(3, 5), (4, 1)])
def test_parallel_cycle_limit_is_total(tmp_settings_file: Path, workers: int, limit: int) -> None:
    s = load_settings(tmp_settings_file)
    with mock.patch("factory.loop.run_cycle") as cycle, \
         mock.patch("factory.loop._start_discovery_worker", return_value=threading.Event()):
        assert run_loop(s, rng=random.Random(0), workers=workers, max_cycles_override=limit) == limit
    assert cycle.call_count == limit


def test_parallel_workers_finish_inflight_cycle_on_shutdown(tmp_settings_file: Path) -> None:
    from factory.loop import _ShutdownFlag

    s = load_settings(tmp_settings_file)
    flag = _ShutdownFlag()
    barrier = threading.Barrier(2, timeout=10)

    def cycle(*args, **kwargs):
        barrier.wait()
        flag.set()
        return mock.Mock()

    with mock.patch("factory.loop.run_cycle", side_effect=cycle) as cycles, \
         mock.patch("factory.loop._start_discovery_worker", return_value=threading.Event()):
        assert run_loop(s, rng=random.Random(0), workers=2,
                        max_cycles_override=0, shutdown_flag=flag) == 2
    assert cycles.call_count == 2


def test_parallel_login_failure_stops_all_workers(tmp_settings_file: Path) -> None:
    from factory.generate import GeneratorUnavailable
    from factory.loop import _ShutdownFlag

    s = load_settings(tmp_settings_file)
    flag = _ShutdownFlag()
    barrier = threading.Barrier(2, timeout=10)

    def cycle(worker, **kwargs):
        barrier.wait()
        if worker.node_id == "local-w2":
            raise GeneratorUnavailable("log in")
        assert flag._event.wait(5)
        return mock.Mock()

    with mock.patch("factory.loop.run_cycle", side_effect=cycle) as cycles, \
         mock.patch("factory.loop._start_discovery_worker", return_value=threading.Event()), \
         pytest.raises(GeneratorUnavailable, match="log in"):
        run_loop(s, rng=random.Random(0), workers=2,
                 max_cycles_override=0, shutdown_flag=flag)
    assert cycles.call_count == 2


def test_parallel_workers_reject_distributed_sync(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    s = replace(s, sync=replace(s.sync, enabled=True))
    with mock.patch("factory.loop.run_cycle") as cycle, \
         pytest.raises(ValueError, match="separate checkouts"):
        run_loop(s, rng=random.Random(0), workers=2)
    cycle.assert_not_called()


def test_worker_startup_failure_stops_running_peers(tmp_settings_file: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from factory.loop import _ShutdownFlag

    s = load_settings(tmp_settings_file)
    flag = _ShutdownFlag()
    started = threading.Event()
    submit = ThreadPoolExecutor.submit

    def cycle(*args, **kwargs):
        started.set()
        assert flag._event.wait(5)
        return mock.Mock()

    def start(pool, *args, **kwargs):
        if started.is_set():
            raise RuntimeError("cannot start worker")
        future = submit(pool, *args, **kwargs)
        assert started.wait(5)
        return future

    with mock.patch("factory.loop.run_cycle", side_effect=cycle), \
         mock.patch("factory.loop._start_discovery_worker", return_value=threading.Event()), \
         mock.patch.object(ThreadPoolExecutor, "submit", start), \
         pytest.raises(RuntimeError, match="cannot start worker"):
        run_loop(s, rng=random.Random(0), workers=2,
                 max_cycles_override=0, shutdown_flag=flag)
    assert flag.is_set()


def test_main_passes_workers(tmp_settings_file: Path) -> None:
    with mock.patch("factory.loop.run_loop") as loop, \
         mock.patch("factory.loop.configure_logging"):
        assert main(["--settings", str(tmp_settings_file), "--workers", "3"]) == 0
    assert loop.call_args.kwargs["workers"] == 3


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_main_rejects_invalid_workers(tmp_settings_file: Path, workers: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--settings", str(tmp_settings_file), "--workers", workers])
    assert exc.value.code == 2
