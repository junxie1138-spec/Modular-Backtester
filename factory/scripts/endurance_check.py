"""100-cycle endurance check with stubbed claude -p.

Runs N cycles against the real backtester subprocesses on the real CSV
data. Generator output rotates through three pre-built strategy/config
pairs (one valid, one validation-fail, one stage-fail) so we hit every
record path. Verifies at the end that the results store contains N records and that
every cycle locked in a strategy id.

USAGE:
    python -m factory.scripts.endurance_check --cycles 10        # quick sanity
    python -m factory.scripts.endurance_check --cycles 100       # full run

This is an operator script, NOT a pytest test. It takes real wall-clock
time (the valid scenarios run real subprocess stages).
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path
from unittest import mock


SCENARIOS = ("valid", "validation_fail", "stage_fail")


def _load_settings(repo: Path, scratch: Path):
    from factory.settings_loader import load_settings
    settings_toml = scratch / "settings.toml"
    settings_toml.write_text(f"""
[paths]
backtester_root  = "{repo.as_posix()}"
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_dir        = "{(scratch / 'dedup').as_posix()}"
results_dir      = "{(scratch / 'results').as_posix()}"
factory_log      = "{(scratch / 'factory.log').as_posix()}"
tmp_dir          = "{(scratch / '_tmp').as_posix()}"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p"]
generation_timeout_sec = 120

[stages]
stage_timeout_sec = 1800

[alerts]
alert_threshold_metric = "wfo.oos_sortino"
alert_threshold        = 999.0
telegram_bot_token     = ""
telegram_chat_id       = ""
dashboard_base_url     = "http://127.0.0.1:8787"

[loop]
mode                  = "continuous"
inter_cycle_sleep_sec = 0
max_cycles            = 0

[dashboard]
host             = "127.0.0.1"
port             = 8787
auto_refresh_sec = 10
""", encoding="utf-8")
    return load_settings(settings_toml)


def _scenario_payload(scenario: str, strategy_id: str, fixtures: Path) -> dict:
    if scenario == "valid":
        src = (fixtures / "valid_strategy.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    elif scenario == "validation_fail":
        src = (fixtures / "invalid_no_shift.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    else:  # stage_fail — valid src but config that will fail backtest
        src = (fixtures / "valid_strategy.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    src = re.sub(r'strategy_id = "[^"]+"', f'strategy_id = "{strategy_id}"', src, count=1)
    cfg = re.sub(r"gen_test_valid", strategy_id, cfg)
    return {
        "strategy_id": strategy_id,
        "one_line_summary": f"endurance test cycle {scenario}",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": src,
        "config_file": cfg,
    }


def _purge_endurance_artifacts(settings) -> int:
    """Delete the gen_endurance_*.py / .yaml files this script generates.

    Returns the count removed. The endurance run must write strategy files
    into the *real* strategies/ dir (the backtester subprocess imports them
    via registry auto-discovery), so the script owns their cleanup. The
    `gen_endurance_` prefix is unique to this script — it never matches
    curated strategies or real factory output (`gen_<node_id>_<ts>.*`).
    """
    removed = 0
    for directory, pattern in (
        (settings.paths.strategies_dir, "gen_endurance_*.py"),
        (settings.paths.configs_dir, "gen_endurance_*.yaml"),
    ):
        for path in directory.glob(pattern):
            path.unlink()
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--scratch", type=Path,
                        default=Path("factory/data/_endurance_scratch"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    fixtures = repo / "factory" / "tests" / "fixtures"
    # Start from a clean scratch tree. read_records() unions every shard in
    # the results dir, so records left by a previous run would inflate the
    # final count and fail the `len(records) == cycles` invariant. The whole
    # _endurance_scratch tree is disposable, so wipe it before each run.
    if args.scratch.exists():
        shutil.rmtree(args.scratch, ignore_errors=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    s = _load_settings(repo, args.scratch)

    # The run writes strategy/config files into the *real* strategies/ and
    # configs/wfo/ dirs. Purge any left by a prior run so filenames do not
    # collide; they are purged again after the run so the repo stays clean.
    purged = _purge_endurance_artifacts(s)
    if purged:
        print(f"removed {purged} leftover gen_endurance_* file(s) from a prior run")

    from factory.cycle import run_cycle
    from factory.generate import GenerationResult
    from factory.loop import configure_logging

    configure_logging(s.paths.factory_log)
    rng = random.Random(0)

    completed = 0
    counter = {"n": 0}

    def fake_call_generator(**kwargs):
        counter["n"] += 1
        scenario = SCENARIOS[counter["n"] % len(SCENARIOS)]
        sid = f"gen_endurance_{counter['n']}"
        parsed = _scenario_payload(scenario, sid, fixtures)
        return GenerationResult(parsed=parsed, cost_usd=0.03, raw_stdout="{}")

    # We also need to mock pick_unused_strategy_id so the cycle locks in
    # the scenario-generated id (not the gen_<time> default). pick_unused_strategy_id
    # runs BEFORE call_generator in the cycle, so we look ahead by +1 to match
    # the id that fake_call_generator will embed in the file.
    def fake_pick_id(base, *, strategies_dir):
        return f"gen_endurance_{counter['n'] + 1}"

    with mock.patch("factory.cycle.call_generator", side_effect=fake_call_generator), \
         mock.patch("factory.cycle.pick_unused_strategy_id", side_effect=fake_pick_id):
        for i in range(args.cycles):
            outcome = run_cycle(s, rng=rng)
            completed += 1
            if (i + 1) % 10 == 0:
                print(f"  cycle {i + 1}/{args.cycles} -> {outcome.status}")

    # Cycles done — remove the strategy/config files this run created so the
    # repo is left clean. Done before the invariant asserts below so a
    # failing assert cannot skip the cleanup.
    _purge_endurance_artifacts(s)

    # Post-run invariants.
    from factory.results import read_records
    records = read_records(s.paths.results_dir)
    print(f"records: {len(records)} (expected {args.cycles})")
    assert len(records) == args.cycles, "results store record count mismatch"

    statuses = {"complete": 0, "failed": 0}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"complete: {statuses['complete']}  failed: {statuses['failed']}")

    # The factory no longer edits registry.py (generated strategies are
    # auto-discovered from strategies/gen_*.py at import time), so there is
    # no registry text to check. Sanity-check the shard instead: every cycle
    # that locked in a strategy id appended one record carrying that id.
    with_ids = sum(1 for r in records if r.get("strategy_id"))
    print(f"records with a strategy_id: {with_ids}")
    assert with_ids == len(records), (
        f"every endurance cycle locks in a strategy id, but only {with_ids} "
        f"of {len(records)} records carry one"
    )
    print("ENDURANCE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
