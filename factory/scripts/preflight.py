"""Preflight readiness check for the distributed strategy factory.

Verifies that a machine can participate in the distributed factory pool.
It runs a sequence of independent checks and prints a PASS / WARN / FAIL
line for each, then a final verdict:

- Python version           (>= 3.11)
- Factory dependencies      (backtester package, pandas/numpy/pyyaml,
                             Flask; yfinance is optional)
- Settings load             (settings.toml + settings.local.toml parse)
- node_id                   (set, and not the single-machine default)
- Data directories          (results / dedup / log / tmp are writable)
- [sync] config             (distributed mode enabled)
- claude CLI                (resolvable on PATH and - unless
                             --skip-claude-probe - authenticated, via a
                             trivial live `claude -p` call)
- git version               (>= 2.28, required by factory/sync.py)
- Remote access             (the [sync] remote is reachable and the
                             stored credentials authenticate without a
                             prompt)

USAGE:
    python -m factory.scripts.preflight
    python -m factory.scripts.preflight --skip-claude-probe   # no token spend
    python -m factory.scripts.preflight --skip-remote         # offline
    python -m factory.scripts.preflight --settings path/to/settings.toml

Exit code is 0 when no check FAILs (WARNs are allowed) and 1 otherwise,
so the script is usable as an unattended gate.

This is an operator script, NOT a pytest test.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

MIN_PYTHON = (3, 11)
MIN_GIT = (2, 28)
# A trivial "ok" generation returns in seconds; cap the probe well short of
# the factory's real generation_timeout so a hung CLI fails fast.
_CLAUDE_PROBE_TIMEOUT_SEC = 120


def _check_python() -> tuple[str, str]:
    v = sys.version_info
    cur = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        return PASS, f"Python {cur}"
    return FAIL, f"Python {cur} - factory needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"


def _check_dependencies() -> tuple[str, str]:
    missing: list[str] = []
    for mod, label in (
        ("backtester", "backtester (run: pip install -e .)"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("flask", "flask (dashboard)"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)
    optional_missing: list[str] = []
    try:
        __import__("yfinance")
    except ImportError:
        optional_missing.append("yfinance")
    if missing:
        return FAIL, "missing: " + ", ".join(missing)
    if optional_missing:
        return WARN, (
            "core deps present; optional missing: "
            + ", ".join(optional_missing)
            + " - needed only for held-out promotion's yfinance tickers"
        )
    return PASS, "all factory dependencies importable"


def _check_settings(settings_path: Path) -> tuple[str, str, Optional[object]]:
    """Returns (status, detail, settings_or_None)."""
    try:
        from factory.settings_loader import load_settings
        s = load_settings(settings_path)
    except FileNotFoundError:
        return FAIL, f"settings file not found: {settings_path}", None
    except Exception as exc:  # malformed TOML, invalid node_id, missing key
        return FAIL, f"settings failed to load: {exc}", None
    return PASS, f"loaded {settings_path}", s


def _check_node_id(settings) -> tuple[str, str]:
    # settings_loader already validated node_id against ^[a-z0-9][a-z0-9-]*$;
    # reaching here means the value is structurally valid.
    nid = settings.node_id
    if nid == "local":
        return WARN, (
            'node_id is the default "local" - every machine in a distributed '
            'pool needs a unique node_id (set it in settings.local.toml, '
            'e.g. "desk", "vps1")'
        )
    return PASS, f'node_id = "{nid}"'


def _check_writable(settings) -> tuple[str, str]:
    p = settings.paths
    for d in (p.results_dir, p.dedup_dir, p.factory_log.parent, p.tmp_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".preflight_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return FAIL, f"cannot write to {d}: {exc}"
    return PASS, "results / dedup / log / tmp directories are writable"


def _check_sync_enabled(settings) -> tuple[str, str]:
    sy = settings.sync
    if sy.enabled:
        return PASS, f"[sync] enabled - branch={sy.branch!r} remote={sy.remote!r}"
    return WARN, (
        "[sync] enabled = false - this machine runs single-machine only; "
        "set enabled = true in settings.toml to join the pool"
    )


def _check_claude(settings, *, probe: bool) -> tuple[str, str]:
    cmd = settings.generation.claude_cmd if settings else "claude"
    resolved = shutil.which(cmd)
    if resolved is None:
        return FAIL, f"claude CLI not found on PATH (claude_cmd={cmd!r})"
    if not probe:
        return WARN, f"claude resolved at {resolved}; live auth probe skipped (--skip-claude-probe)"
    flags = (
        list(settings.generation.claude_flags)
        if settings else ["-p", "--output-format", "json"]
    )
    try:
        proc = subprocess.run(
            [resolved, *flags],
            input="Reply with exactly the single word: ok",
            capture_output=True, text=True, encoding="utf-8",
            timeout=_CLAUDE_PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return FAIL, f"claude probe timed out after {_CLAUDE_PROBE_TIMEOUT_SEC}s"
    except OSError as exc:
        return FAIL, f"claude probe failed to start: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        return FAIL, f"claude exited {proc.returncode}: {tail}"
    out = proc.stdout.strip()
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError:
        if out:
            return WARN, "claude responded, but output was not a single JSON object - likely OK"
        return FAIL, "claude produced no output"
    if isinstance(envelope, dict) and envelope.get("is_error"):
        return FAIL, f"claude returned an error envelope: {envelope.get('result') or envelope}"
    if isinstance(envelope, dict) and "result" in envelope:
        return PASS, "claude CLI authenticated and responding"
    return WARN, "claude responded, but the JSON envelope shape was unexpected"


def _check_git_version() -> tuple[str, str]:
    git = shutil.which("git")
    if git is None:
        return FAIL, "git not found on PATH"
    try:
        out = subprocess.run(
            [git, "--version"], capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FAIL, f"git --version failed: {exc}"
    m = re.search(r"(\d+)\.(\d+)", out)
    if not m:
        return WARN, f"could not parse a version from {out.strip()!r}"
    ver = (int(m.group(1)), int(m.group(2)))
    if ver >= MIN_GIT:
        return PASS, out.strip()
    return FAIL, (
        f"{out.strip()} - distributed sync needs git >= {MIN_GIT[0]}.{MIN_GIT[1]}"
    )


def _check_remote(settings) -> tuple[str, str]:
    git = shutil.which("git")
    if git is None:
        return FAIL, "git not found on PATH"
    remote = settings.sync.remote if settings else "origin"
    branch = settings.sync.branch if settings else "factory-pool"
    cwd = settings.paths.backtester_root if settings else Path.cwd()
    # GIT_TERMINAL_PROMPT=0 turns a would-be credential prompt into an
    # immediate failure - the same prompt would silently hang the
    # unattended loop's sync_push, so we want it surfaced here, not waited on.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            [git, "ls-remote", "--heads", remote],
            cwd=str(cwd), capture_output=True, text=True, timeout=60, env=env,
        )
    except subprocess.TimeoutExpired:
        return FAIL, f"git ls-remote {remote} timed out (network or auth hang)"
    except OSError as exc:
        return FAIL, f"git ls-remote failed to start: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-300:]
        return FAIL, (
            f"cannot reach/authenticate remote {remote!r}: {tail} - the "
            "unattended loop needs non-interactive credentials (an SSH key "
            "or a cached credential helper)"
        )
    has_pool = any(
        line.endswith(f"refs/heads/{branch}")
        for line in proc.stdout.splitlines()
    )
    note = (
        f"pool branch {branch!r} already exists"
        if has_pool
        else f"pool branch {branch!r} not yet created (the first machine publishes it)"
    )
    return PASS, f"remote {remote!r} reachable and authenticated; {note}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        "factory.scripts.preflight",
        description="Check whether this machine can run the distributed factory.",
    )
    parser.add_argument(
        "--settings", type=Path, default=Path("factory/config/settings.toml"),
        help="path to settings.toml (default: factory/config/settings.toml)",
    )
    parser.add_argument(
        "--skip-claude-probe", action="store_true",
        help="skip the live `claude -p` auth probe (no token spend)",
    )
    parser.add_argument(
        "--skip-remote", action="store_true",
        help="skip the git remote reachability/auth check (offline)",
    )
    args = parser.parse_args(argv)

    results: list[tuple[str, str, str]] = []  # (name, status, detail)

    def record(name: str, status_detail: tuple[str, str]) -> None:
        results.append((name, status_detail[0], status_detail[1]))

    record("Python version", _check_python())
    record("Factory dependencies", _check_dependencies())

    st_status, st_detail, settings = _check_settings(args.settings)
    results.append(("Settings load", st_status, st_detail))

    if settings is not None:
        record("node_id", _check_node_id(settings))
        record("Data directories", _check_writable(settings))
        record("[sync] config", _check_sync_enabled(settings))

    record("claude CLI", _check_claude(settings, probe=not args.skip_claude_probe))
    record("git version", _check_git_version())
    if args.skip_remote:
        results.append(("Remote access", WARN, "skipped (--skip-remote)"))
    else:
        record("Remote access", _check_remote(settings))

    width = max(len(name) for name, _, _ in results)
    print()
    print("  Distributed factory - preflight check")
    print("  " + "-" * 52)
    for name, status, detail in results:
        print(f"  [{status}] {name.ljust(width)}  {detail}")
    print()

    fails = [n for n, s, _ in results if s == FAIL]
    warns = [n for n, s, _ in results if s == WARN]
    if fails:
        print(f"  VERDICT: NOT READY - {len(fails)} check(s) failed: {', '.join(fails)}")
        return 1
    if warns:
        print(f"  VERDICT: READY - with {len(warns)} warning(s); review the lines above.")
        return 0
    print("  VERDICT: READY - this machine can join the distributed factory pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
