"""Git coordination for the distributed strategy factory.

Three operations wrap `git` via subprocess:

- bootstrap()  — one-time, idempotent: ensure the `factory-pool` branch
                 exists (create off master + publish on first run, or track
                 an existing remote branch), ensure scratch dirs are
                 gitignored, and fold any legacy single-file stores into
                 this machine's shards.
- sync_pull()  — before a cycle: rebase the pool onto this machine. Skips
                 the sync (does not rebase) if the working tree has
                 unexpected tracked changes.
- sync_push()  — after a cycle: commit + push this machine's new files.
                 No-op when nothing is staged; retries a non-fast-forward
                 rejection up to push_retries times.

All three are no-ops when [sync] enabled = false. Every git failure raises
SyncError; the loop (factory/loop.py) wraps each call so a failure is logged
and generation continues — sync failure never aborts the factory.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from factory.settings_loader import Settings

log = logging.getLogger(__name__)

# Local-scratch paths that must never reach the shared pool branch.
_SCRATCH_GITIGNORE_ENTRIES = (
    "factory/data/_tmp/",
    "factory/logs/",
    "output/runs/",
)


class SyncError(RuntimeError):
    """A git operation failed during sync."""


def _git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run `git <args>` in `cwd`. Raises SyncError on failure when `check`."""
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True, encoding="utf-8",
    )
    if check and proc.returncode != 0:
        raise SyncError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    return proc


def _branch_exists_local(branch: str, *, root: Path) -> bool:
    return _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=root, check=False).returncode == 0


def _current_branch(root: Path) -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root).stdout.strip()


def _tracked_dirt(root: Path) -> list[str]:
    """Return porcelain lines for tracked changes (modified/staged/deleted).

    Untracked files (status `??`) are excluded — uniquely-named generated
    files never block a rebase, so they are not "dirt".
    """
    proc = _git(["status", "--porcelain"], cwd=root)
    return [
        line for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("??")
    ]


def _ensure_gitignore(root: Path) -> None:
    """Append any missing local-scratch entries to .gitignore. Idempotent."""
    gi = root / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    present = {line.strip() for line in text.splitlines()}
    missing = [e for e in _SCRATCH_GITIGNORE_ENTRIES if e not in present]
    if not missing:
        return
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n".join(missing) + "\n"
    gi.write_text(text, encoding="utf-8")
    log.info("sync bootstrap: added .gitignore entries %s", missing)


def _fold_legacy_stores(settings: Settings) -> None:
    """Copy a pre-existing single-file results.json / dedup_log.txt into this
    machine's shard. Idempotent: skips when the shard already exists.
    """
    root = settings.paths.backtester_root
    node_id = settings.node_id
    legacy_results = root / "factory" / "data" / "results.json"
    shard_results = settings.paths.results_dir / f"{node_id}.jsonl"
    if legacy_results.exists() and not shard_results.exists():
        shard_results.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_results, shard_results)
        log.info("sync bootstrap: folded legacy results.json -> %s", shard_results)
    legacy_dedup = root / "factory" / "data" / "dedup_log.txt"
    shard_dedup = settings.paths.dedup_dir / f"{node_id}.txt"
    if legacy_dedup.exists() and not shard_dedup.exists():
        shard_dedup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_dedup, shard_dedup)
        log.info("sync bootstrap: folded legacy dedup_log.txt -> %s", shard_dedup)


def _commit_paths(settings: Settings) -> list[str]:
    """The directories this machine commits — its strategies, configs, and
    result/dedup shards, as repo-relative posix pathspecs. Only existing
    paths are returned (git add of a missing pathspec errors)."""
    p = settings.paths
    root = p.backtester_root
    dirs = [p.strategies_dir, p.configs_dir, p.results_dir, p.dedup_dir]
    return [d.relative_to(root).as_posix() for d in dirs if d.exists()]


def bootstrap(settings: Settings) -> None:
    """One-time, idempotent distributed-sync setup. No-op when disabled."""
    if not settings.sync.enabled:
        return
    root = settings.paths.backtester_root
    branch = settings.sync.branch
    remote = settings.sync.remote

    _ensure_gitignore(root)
    _fold_legacy_stores(settings)

    if _branch_exists_local(branch, root=root):
        log.info("sync bootstrap: branch %s already present locally", branch)
        return
    # Not local — see if another machine already published it.
    _git(["fetch", remote, branch], cwd=root, check=False)
    on_remote = _git(["rev-parse", "--verify", "--quiet",
                      f"refs/remotes/{remote}/{branch}"], cwd=root, check=False)
    if on_remote.returncode == 0:
        _git(["branch", "--track", branch, f"{remote}/{branch}"], cwd=root)
        log.info("sync bootstrap: tracking existing remote branch %s", branch)
        return
    # Brand new: create off master and publish. Publishing is intentional
    # remote-mutating behavior — the pool cannot function until the branch
    # is visible to the other machines.
    _git(["branch", branch, "master"], cwd=root)
    _git(["push", "-u", remote, branch], cwd=root)
    log.info("sync bootstrap: created and published branch %s", branch)


def sync_pull(settings: Settings) -> None:
    """Pull the pool before a cycle. No-op when disabled. Skips on a dirty tree."""
    if not settings.sync.enabled:
        return
    root = settings.paths.backtester_root
    branch = settings.sync.branch
    remote = settings.sync.remote

    dirt = _tracked_dirt(root)
    if dirt:
        log.warning("sync_pull: working tree has tracked changes; skipping "
                    "sync this cycle: %s", dirt)
        return
    if _current_branch(root) != branch:
        _git(["checkout", branch], cwd=root)
    _git(["fetch", remote, branch], cwd=root)
    _git(["pull", "--rebase", remote, branch], cwd=root)
    log.info("sync_pull: rebased onto %s/%s", remote, branch)


def sync_push(settings: Settings) -> None:
    """Commit + push this cycle's output. No-op when disabled or nothing staged.

    A non-fast-forward rejection is handled by `git pull --rebase` + retry,
    bounded by push_retries. Because every machine writes only its own
    uniquely-named files and shards, the rebase is always conflict-free.
    """
    if not settings.sync.enabled:
        return
    root = settings.paths.backtester_root
    branch = settings.sync.branch
    remote = settings.sync.remote

    add_paths = _commit_paths(settings)
    if add_paths:
        _git(["add", "--", *add_paths], cwd=root)
    staged = _git(["diff", "--cached", "--quiet"], cwd=root, check=False)
    if staged.returncode == 0:
        log.info("sync_push: nothing staged; no-op")
        return
    _git(["commit", "-m", f"factory({settings.node_id}): pool update"], cwd=root)
    for attempt in range(1, settings.sync.push_retries + 1):
        push = _git(["push", remote, branch], cwd=root, check=False)
        if push.returncode == 0:
            log.info("sync_push: pushed (attempt %d)", attempt)
            return
        log.warning("sync_push: push rejected (attempt %d/%d): %s",
                    attempt, settings.sync.push_retries,
                    (push.stderr or "").strip())
        if attempt < settings.sync.push_retries:
            _git(["pull", "--rebase", remote, branch], cwd=root)
    raise SyncError(
        f"sync_push: push still failing after {settings.sync.push_retries} retries"
    )
