"""Offline tests for factory/sync.py — no network, no real GitHub.

Each test scaffolds throwaway local git repos: a bare repo acts as the
"remote", and one or two clones act as factory machines (nodes).
"""
from __future__ import annotations

import random
import subprocess
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from factory.cycle import CycleOutcome
from factory.loop import run_loop
from factory.settings_loader import load_settings
from factory.sync import SyncError, bootstrap, sync_pull, sync_push, _fold_legacy_stores


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True,
                   capture_output=True, text=True)


def _init_bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "--bare", "-b", "master"], path)
    return path


def _clone(remote: Path, dest: Path) -> Path:
    subprocess.run(["git", "clone", str(remote), str(dest)],
                   check=True, capture_output=True, text=True)
    _git(["config", "user.email", "node@example.com"], dest)
    _git(["config", "user.name", "Node"], dest)
    return dest


def _seed_master(repo: Path) -> None:
    """Put an initial commit on master so factory-pool can branch off it."""
    (repo / "README.md").write_text("factory repo\n", encoding="utf-8")
    (repo / "configs").mkdir(exist_ok=True)
    (repo / "configs" / "wfo").mkdir(exist_ok=True)
    (repo / "configs" / "wfo" / ".gitkeep").write_text("", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)
    _git(["push", "origin", "master"], repo)


_SETTINGS_TEMPLATE = """\
node_id = "{node_id}"

[paths]
backtester_root  = "{root}"
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_dir        = "factory/data/dedup"
results_dir      = "factory/data/results"
factory_log      = "factory/logs/factory.log"
tmp_dir          = "factory/data/_tmp"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p"]
generation_timeout_sec = 60

[stages]
stage_timeout_sec = 300

[alerts]
alert_threshold_metric = "wfo.oos_sharpe"
alert_threshold        = 1.0
telegram_bot_token     = ""
telegram_chat_id       = ""
dashboard_base_url     = "http://127.0.0.1:8787"

[loop]
mode                  = "continuous"
inter_cycle_sleep_sec = 0
max_cycles            = 1

[dashboard]
host             = "127.0.0.1"
port             = 8787
auto_refresh_sec = 10

[sync]
enabled      = {enabled}
branch       = "factory-pool"
remote       = "{remote}"
push_retries = 5
"""


def _node_settings(repo: Path, node_id: str, *, enabled: bool = True,
                    remote: str = "origin"):
    """Write a settings.toml into `repo` and load it."""
    toml = _SETTINGS_TEMPLATE.format(
        node_id=node_id, root=repo.as_posix(),
        enabled="true" if enabled else "false", remote=remote,
    )
    p = repo / "settings.toml"
    p.write_text(textwrap.dedent(toml), encoding="utf-8")
    return load_settings(p)


def _produce_strategy(repo: Path, node_id: str, ts: int) -> str:
    """Simulate a cycle: write a uniquely-named strategy file + a results
    shard line. Returns the strategy id."""
    sid = f"gen_{node_id}_{ts}"
    strat_dir = repo / "strategies"
    strat_dir.mkdir(parents=True, exist_ok=True)
    (strat_dir / f"{sid}.py").write_text(f"# {sid}\n", encoding="utf-8")
    results_dir = repo / "factory" / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{node_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(f'{{"strategy_id": "{sid}"}}\n')
    return sid


# --------------------------------------------------------------------------


def test_disabled_sync_is_a_noop(tmp_path: Path) -> None:
    repo = _clone(_init_bare_remote(tmp_path / "remote.git"), tmp_path / "node")
    _seed_master(repo)
    s = _node_settings(repo, "local", enabled=False)
    # None of these should touch git or raise.
    bootstrap(s)
    sync_pull(s)
    sync_push(s)
    assert subprocess.run(["git", "rev-parse", "--verify", "factory-pool"],
                          cwd=str(repo), capture_output=True).returncode != 0


def test_bootstrap_creates_and_publishes_branch(tmp_path: Path) -> None:
    remote = _init_bare_remote(tmp_path / "remote.git")
    repo = _clone(remote, tmp_path / "node")
    _seed_master(repo)
    s = _node_settings(repo, "desk")
    bootstrap(s)
    # Branch exists locally and on the remote.
    assert subprocess.run(["git", "rev-parse", "--verify", "factory-pool"],
                          cwd=str(repo), capture_output=True).returncode == 0
    ls = subprocess.run(["git", "ls-remote", "--heads", "origin", "factory-pool"],
                        cwd=str(repo), capture_output=True, text=True)
    assert "factory-pool" in ls.stdout
    # Idempotent.
    bootstrap(s)


def test_bootstrap_second_node_tracks_existing_branch(tmp_path: Path) -> None:
    remote = _init_bare_remote(tmp_path / "remote.git")
    node_a = _clone(remote, tmp_path / "a")
    _seed_master(node_a)
    bootstrap(_node_settings(node_a, "a"))
    # Second machine: branch already on the remote, not local.
    node_b = _clone(remote, tmp_path / "b")
    bootstrap(_node_settings(node_b, "b"))
    assert subprocess.run(["git", "rev-parse", "--verify", "factory-pool"],
                          cwd=str(node_b), capture_output=True).returncode == 0


def test_fold_legacy_stores(tmp_path: Path) -> None:
    repo = _clone(_init_bare_remote(tmp_path / "remote.git"), tmp_path / "node")
    _seed_master(repo)
    legacy_data = repo / "factory" / "data"
    legacy_data.mkdir(parents=True, exist_ok=True)
    (legacy_data / "results.json").write_text('{"strategy_id":"old"}\n', encoding="utf-8")
    (legacy_data / "dedup_log.txt").write_text("an old idea no tab\n", encoding="utf-8")
    s = _node_settings(repo, "desk")
    _fold_legacy_stores(s)
    assert (legacy_data / "results" / "desk.jsonl").read_text(encoding="utf-8") \
        == '{"strategy_id":"old"}\n'
    assert (legacy_data / "dedup" / "desk.txt").read_text(encoding="utf-8") \
        == "an old idea no tab\n"
    # Idempotent: a second fold does not clobber the shard.
    (legacy_data / "results" / "desk.jsonl").write_text("MUTATED\n", encoding="utf-8")
    _fold_legacy_stores(s)
    assert (legacy_data / "results" / "desk.jsonl").read_text(encoding="utf-8") == "MUTATED\n"


def test_two_nodes_converge_conflict_free(tmp_path: Path) -> None:
    remote = _init_bare_remote(tmp_path / "remote.git")
    node_a = _clone(remote, tmp_path / "a")
    _seed_master(node_a)
    sa = _node_settings(node_a, "a")
    bootstrap(sa)

    node_b = _clone(remote, tmp_path / "b")
    sb = _node_settings(node_b, "b")
    bootstrap(sb)

    # Node A produces and pushes.
    sync_pull(sa)
    _produce_strategy(node_a, "a", 1000)
    sync_push(sa)

    # Node B pulls A's work, produces its own.
    sync_pull(sb)
    _produce_strategy(node_b, "b", 2000)

    # Node A pushes again FIRST -> the remote moves ahead of B.
    sync_pull(sa)
    _produce_strategy(node_a, "a", 1001)
    sync_push(sa)

    # Node B's push is now a non-fast-forward: sync_push must rebase + retry
    # and converge.
    sync_push(sb)

    # Node A pulls -> sees the whole pool.
    sync_pull(sa)
    assert (node_a / "strategies" / "gen_a_1000.py").exists()
    assert (node_a / "strategies" / "gen_a_1001.py").exists()
    assert (node_a / "strategies" / "gen_b_2000.py").exists()
    assert (node_a / "factory" / "data" / "results" / "a.jsonl").exists()
    assert (node_a / "factory" / "data" / "results" / "b.jsonl").exists()


def test_sync_push_noop_when_nothing_changed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repo = _clone(_init_bare_remote(tmp_path / "remote.git"), tmp_path / "node")
    _seed_master(repo)
    s = _node_settings(repo, "desk")
    bootstrap(s)
    sync_pull(s)
    with caplog.at_level("INFO"):
        sync_push(s)   # nothing produced this cycle
    assert "no-op" in caplog.text.lower()


def test_sync_pull_skips_on_dirty_tree(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repo = _clone(_init_bare_remote(tmp_path / "remote.git"), tmp_path / "node")
    _seed_master(repo)
    s = _node_settings(repo, "desk")
    bootstrap(s)
    # Make a tracked file dirty.
    (repo / "README.md").write_text("dirtied\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        sync_pull(s)   # must not raise
    assert "skipping" in caplog.text.lower()


def test_sync_pull_raises_on_unreachable_remote(tmp_path: Path) -> None:
    repo = tmp_path / "lonely"
    repo.mkdir()
    _git(["init", "-b", "master"], repo)
    _git(["config", "user.email", "n@example.com"], repo)
    _git(["config", "user.name", "N"], repo)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)
    _git(["branch", "factory-pool"], repo)
    _git(["remote", "add", "origin", str(tmp_path / "does_not_exist.git")], repo)
    s = _node_settings(repo, "desk")
    with pytest.raises(SyncError):
        sync_pull(s)


def test_run_loop_swallows_sync_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """With sync enabled but the remote unreachable, run_loop calls the sync
    hooks, logs each failure, and still completes its cycles — sync failure
    never aborts the loop."""
    repo = tmp_path / "node"
    repo.mkdir()
    _git(["init", "-b", "master"], repo)
    _git(["config", "user.email", "n@example.com"], repo)
    _git(["config", "user.name", "N"], repo)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)
    _git(["remote", "add", "origin", str(tmp_path / "missing.git")], repo)
    s = _node_settings(repo, "desk")   # sync enabled, remote unreachable

    fake = CycleOutcome(status="failed", failed_stage="generation",
                        strategy_id=None, record={"status": "failed"})
    with mock.patch("factory.loop.run_cycle", return_value=fake) as rc, \
         mock.patch("factory.loop.sync_push", wraps=sync_push) as sp, \
         caplog.at_level("ERROR"):
        completed = run_loop(s, rng=random.Random(0), max_cycles_override=1)

    # The cycle ran and the loop returned normally — no SyncError escaped.
    assert rc.call_count == 1
    assert completed == 1
    # sync_push was wired into the loop and invoked once.
    assert sp.call_count == 1
    # bootstrap and sync_pull both hit the unreachable remote, raised
    # SyncError, and were caught + logged by run_loop (not propagated).
    assert "sync bootstrap failed" in caplog.text
    assert "sync_pull failed" in caplog.text
