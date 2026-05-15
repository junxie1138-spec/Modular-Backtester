# Distributed Strategy Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let multiple machines each run the strategy factory and contribute generated strategies into one shared git-coordinated pool, with no two machines ever writing the same file.

**Architecture:** Every machine has a stable `node_id`. All machine-owned state is keyed by `node_id`: strategy IDs become `gen_<node_id>_<ts>`, the results and dedup stores become per-machine shard directories, and the registry stops being edited (auto-discovery replaces it). A new `factory/sync.py` wraps `git` to pull/push the shared `factory-pool` branch around each cycle. Because every file has exactly one writer, `git pull --rebase` is always conflict-free.

**Tech Stack:** Python 3.11+ (`tomllib`, `importlib`, `subprocess`), pytest, git.

---

## Background for the implementing engineer

The factory (`factory/`) is a single-machine loop. `factory/loop.py:run_loop` calls `factory/cycle.py:run_cycle` repeatedly. Each cycle: pulls random "slots", builds a prompt, calls Claude, validates the generated strategy, writes it to disk, runs backtest/optimize/WFO stages, and records the outcome.

Three pieces of shared mutable state break when two machines share a repo, and this plan removes all three:
1. `factory/data/results.json` — one JSONL file, appended every cycle → becomes a directory of per-machine `*.jsonl` shards.
2. `factory/data/dedup_log.txt` — one text file, appended every cycle → becomes a directory of per-machine `*.txt` shards (now timestamped).
3. `backtester/strategies/registry.py` — read-modify-written every cycle → the per-strategy edit is removed; the registry auto-discovers `strategies/gen_*.py` at import time.

**Test commands** (run from repo root `C:\Users\aiden\Documents\VScode_Work\Backtester`):
- Factory suite: `python -m pytest factory/tests/ -q`
- Backtester registry suite: `python -m pytest tests/unit/test_strategy_registry.py -q`
- Both: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`

The test suite is expected to be **fully green at the end of every task**. Tasks are ordered so each one leaves a working tree.

**Settings loading:** `factory/settings_loader.py:load_settings` reads `factory/config/settings.toml`, then shallow-merges an optional gitignored `settings.local.toml` over it (per top-level key/section). `factory/tests/conftest.py` provides a `tmp_settings_file` fixture that writes a complete `settings.toml` into `tmp_path`.

---

## Task 1: Root `.gitignore` — local-scratch entries

The factory's local-scratch directories must never be committed to the shared `factory-pool` branch. `output/runs/*` is already ignored at the repo root; `factory/data/_tmp/` and `factory/logs/` are only ignored by `factory/.gitignore` (relative paths). Add explicit repo-root entries so `factory/sync.py:bootstrap` (Task 7) finds them already present.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add the three scratch entries to `.gitignore`**

The current `.gitignore` ends with:

```
output/runs/*
!output/runs/.gitkeep
data/processed/*
!data/processed/.gitkeep
.env
.DS_Store
```

Append three lines at the end of the file so it becomes:

```
output/runs/*
!output/runs/.gitkeep
data/processed/*
!data/processed/.gitkeep
.env
.DS_Store
factory/data/_tmp/
factory/logs/
output/runs/
```

(`output/runs/` is added in addition to the existing `output/runs/*` — git tolerates both, and the exact string `output/runs/` is what `bootstrap`'s gitignore check looks for.)

- [ ] **Step 2: Verify the suite still passes**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS (no behavior changed).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore factory scratch dirs at repo root"
```

---

## Task 2: Settings — `node_id` and `[sync]` section

Add a validated per-machine `node_id` (default `"local"`) and a `[sync]` config section to the settings system. This task is purely additive — no existing field is removed — so the suite stays green.

**Files:**
- Modify: `factory/settings_loader.py`
- Modify: `factory/config/settings.toml`
- Test: `factory/tests/test_settings_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `factory/tests/test_settings_loader.py`:

```python
import pytest


def test_node_id_defaults_to_local(tmp_settings_file: Path) -> None:
    """When no node_id is set anywhere, it defaults to 'local'."""
    s = load_settings(tmp_settings_file)
    assert s.node_id == "local"


def test_node_id_read_from_local_override(tmp_settings_file: Path) -> None:
    """A top-level node_id in settings.local.toml is picked up."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text('node_id = "desk"\n', encoding="utf-8")
    s = load_settings(tmp_settings_file)
    assert s.node_id == "desk"


def test_malformed_node_id_is_fatal(tmp_settings_file: Path) -> None:
    """A node_id that is not ^[a-z0-9][a-z0-9-]*$ fails settings load."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text('node_id = "Bad_ID"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="node_id"):
        load_settings(tmp_settings_file)


def test_sync_section_defaults(tmp_settings_file: Path) -> None:
    """With no [sync] section, sync is disabled with documented defaults."""
    s = load_settings(tmp_settings_file)
    assert s.sync.enabled is False
    assert s.sync.branch == "factory-pool"
    assert s.sync.remote == "origin"
    assert s.sync.push_retries == 5


def test_sync_section_explicit(tmp_settings_file: Path) -> None:
    """An explicit [sync] section overrides the defaults."""
    local = tmp_settings_file.parent / "settings.local.toml"
    local.write_text(
        "[sync]\n"
        "enabled = true\n"
        "branch = \"pool-x\"\n"
        "push_retries = 9\n",
        encoding="utf-8",
    )
    s = load_settings(tmp_settings_file)
    assert s.sync.enabled is True
    assert s.sync.branch == "pool-x"
    assert s.sync.remote == "origin"   # untouched key keeps its default
    assert s.sync.push_retries == 9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest factory/tests/test_settings_loader.py -q`
Expected: FAIL — `Settings` has no attribute `node_id` / `sync`.

- [ ] **Step 3: Add `re` import and the `SyncCfg` dataclass**

In `factory/settings_loader.py`, change the top imports:

```python
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
```

After the `ScreeningCfg` dataclass (around line 69) add:

```python
@dataclass(slots=True, frozen=True)
class SyncCfg:
    enabled: bool
    branch: str
    remote: str
    push_retries: int
```

- [ ] **Step 4: Add `node_id` and `sync` to the `Settings` dataclass**

Change the `Settings` dataclass so it reads:

```python
@dataclass(slots=True, frozen=True)
class Settings:
    node_id: str
    paths: Paths
    generation: GenerationCfg
    stages: StagesCfg
    alerts: AlertsCfg
    loop: LoopCfg
    dashboard: DashboardCfg
    promotion: PromotionCfg
    screening: ScreeningCfg
    sync: SyncCfg
```

- [ ] **Step 5: Add the `node_id` regex constant**

After the imports, before the first dataclass, add:

```python
# node_id is used in filenames and git-safe paths, so it is constrained.
_NODE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
```

- [ ] **Step 6: Read, validate, and build `node_id` + `sync` in `load_settings`**

In `load_settings`, after the `local` merge block and before `p = raw["paths"]`, add:

```python
    node_id = str(raw.get("node_id", "local"))
    if not _NODE_ID_RE.match(node_id):
        raise ValueError(
            f"invalid node_id {node_id!r}: must match ^[a-z0-9][a-z0-9-]*$ "
            f"(lowercase letters, digits and hyphens; not starting with a hyphen). "
            f"Set it in factory/config/settings.local.toml."
        )
```

In the `return Settings(...)` call, add `node_id=node_id,` as the first argument and `sync=...` as the last. The `sync` block — add it just before the `return`, next to where `pr`/`sc` are read:

```python
    sy = raw.get("sync", {}) or {}
```

and inside `Settings(...)`:

```python
    return Settings(
        node_id=node_id,
        paths=paths,
        generation=GenerationCfg(
            ...
        ),
        ...
        screening=ScreeningCfg(
            enabled=bool(sc.get("enabled", False)),
            min_optimize_score=float(sc.get("min_optimize_score", 1.3)),
        ),
        sync=SyncCfg(
            enabled=bool(sy.get("enabled", False)),
            branch=str(sy.get("branch", "factory-pool")),
            remote=str(sy.get("remote", "origin")),
            push_retries=int(sy.get("push_retries", 5)),
        ),
    )
```

- [ ] **Step 7: Add `node_id` and `[sync]` to `settings.toml`**

In `factory/config/settings.toml`, add a top-level `node_id` key as the **very first line** (a bare TOML key must precede any `[table]`):

```toml
node_id = "local"   # per-machine; override in settings.local.toml (e.g. "desk", "vps1")

[paths]
```

Append a `[sync]` section at the end of the file:

```toml
[sync]
enabled      = false          # master switch for distributed multi-machine sync
branch       = "factory-pool" # shared pool branch
remote       = "origin"
push_retries = 5
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS — all tests, including the new five.

- [ ] **Step 9: Commit**

```bash
git add factory/settings_loader.py factory/config/settings.toml factory/tests/test_settings_loader.py
git commit -m "feat(factory): add node_id and [sync] settings"
```

---

## Task 3: Strategy ID scheme — `gen_<node_id>_<ts>`

Strategy IDs are minted in `factory/cycle.py` as `gen_<unix-second>`. Prefix them with `node_id` so the derived strategy `.py` and config `.yaml` filenames are globally unique across machines. `pick_unused_strategy_id` (the same-second `_2`/`_3` fallback) is unchanged — it already takes a `base` and is node-agnostic.

**Files:**
- Modify: `factory/cycle.py:80`
- Test: `factory/tests/test_cycle.py`

- [ ] **Step 1: Write the failing test**

Add to `factory/tests/test_cycle.py`:

```python
def test_cycle_strategy_id_includes_node_id(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    """The minted strategy id is gen_<node_id>_<unix-second>."""
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)
    assert s.node_id == "local"   # fixture default
    fake = _fake_claude_result("placeholder")  # placeholder body -> validation fails
    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000):
        outcome = run_cycle(s, rng=random.Random(0))
    assert outcome.strategy_id == "gen_local_1715800000"
    assert outcome.record["strategy_id"] == "gen_local_1715800000"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest factory/tests/test_cycle.py::test_cycle_strategy_id_includes_node_id -q`
Expected: FAIL — `outcome.strategy_id` is `gen_1715800000`, not `gen_local_1715800000`.

- [ ] **Step 3: Mint the node-scoped id in `cycle.py`**

In `factory/cycle.py`, line 80, change:

```python
    base_strategy_id = f"gen_{_now_unix_int()}"
```

to:

```python
    base_strategy_id = f"gen_{s.node_id}_{_now_unix_int()}"
```

(`s` is already the local alias for `settings` defined on line 75.)

- [ ] **Step 4: Update the existing `gen_1715800000` literals in `test_cycle.py`**

The three full-cycle tests (`test_validation_failure_writes_dedup_but_no_files`, `test_complete_cycle_writes_files_registry_record`, `test_stage_failure_writes_failed_record_keeps_dedup_and_files`, `test_screened_out_skips_wfo_and_promotion`) mock `_now_unix_int` → `1715800000` and expect a strategy id of `gen_1715800000`. With `node_id="local"` the id is now `gen_local_1715800000`.

In `factory/tests/test_cycle.py`, replace every occurrence of the string `gen_1715800000` with `gen_local_1715800000`. (The bare integer `1715800000` passed to `_now_unix_int` `return_value=` has no `gen_` prefix and must stay unchanged — only the `gen_1715800000` token changes.)

- [ ] **Step 5: Run the full cycle suite**

Run: `python -m pytest factory/tests/test_cycle.py -q`
Expected: PASS — all tests including the new one.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add factory/cycle.py factory/tests/test_cycle.py
git commit -m "feat(factory): mint node-scoped strategy ids gen_<node>_<ts>"
```

---

## Task 4: Results store — per-machine shards

Replace the single `factory/data/results.json` with a directory `factory/data/results/`. Each machine appends only to its own shard `<node_id>.jsonl`; reads union every shard. This task renames the `results_store` settings key to `results_dir` and updates every call site in one atomic commit.

**Files:**
- Modify: `factory/results.py` (`write_record`, `read_records`)
- Modify: `factory/settings_loader.py` (`Paths.results_store` → `results_dir`)
- Modify: `factory/config/settings.toml`
- Modify: `factory/tests/conftest.py`
- Modify: `factory/cycle.py` (4 `write_record` call sites)
- Modify: `factory/dashboard/server.py` (4 `read_records` call sites)
- Test: `factory/tests/test_results.py`, `test_settings_loader.py`, `test_dashboard.py`, `test_cycle.py`, `test_integration_failures.py`, `test_integration_smoke.py`

- [ ] **Step 1: Rewrite `factory/tests/test_results.py` storage tests**

Keep the four `build_record` / `build_failed_record` tests (lines 35–95) exactly as they are. Replace the storage tests (everything from `test_write_then_read_roundtrip` to end of file) with:

```python
def test_write_record_creates_node_shard(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "strategy_id": "x"}, node_id="desk")
    assert (d / "desk.jsonl").exists()


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "strategy_id": "x"}, node_id="local")
    write_record(d, {"a": 2, "strategy_id": "y"}, node_id="local")
    write_record(d, {"a": 3, "strategy_id": "z"}, node_id="local")
    assert [r["a"] for r in read_records(d)] == [1, 2, 3]


def test_read_records_unions_shards(tmp_path: Path) -> None:
    d = tmp_path / "results"
    write_record(d, {"a": 1, "timestamp": "2026-05-15T09:00:00Z"}, node_id="desk")
    write_record(d, {"a": 2, "timestamp": "2026-05-15T08:00:00Z"}, node_id="laptop")
    recs = read_records(d)
    assert {r["a"] for r in recs} == {1, 2}
    # Callers that need ordering sort by the record timestamp.
    ordered = sorted(recs, key=lambda r: r["timestamp"])
    assert [r["a"] for r in ordered] == [2, 1]


def test_read_records_handles_missing_dir(tmp_path: Path) -> None:
    assert read_records(tmp_path / "nothing") == []


def test_read_records_skips_blank_lines(tmp_path: Path) -> None:
    d = tmp_path / "results"
    d.mkdir()
    (d / "local.jsonl").write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_records(d) == [{"a": 1}, {"a": 2}]


def test_read_records_raises_on_malformed_line(tmp_path: Path) -> None:
    d = tmp_path / "results"
    d.mkdir()
    (d / "local.jsonl").write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(d)
```

- [ ] **Step 2: Run to verify the storage tests fail**

Run: `python -m pytest factory/tests/test_results.py -q`
Expected: FAIL — `write_record` does not accept `node_id`; `read_records` treats the arg as a file.

- [ ] **Step 3: Rewrite `write_record` and `read_records` in `factory/results.py`**

Replace `write_record` and `read_records` (lines 91–114) with:

```python
def write_record(results_dir: Path, record: Record, *, node_id: str) -> None:
    """Append one JSON object as a single line to this machine's shard.

    The shard is `results_dir/<node_id>.jsonl`. Each machine is the sole
    writer of its own shard, so shards never conflict on a git pull/push.
    """
    shard = results_dir / f"{node_id}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with shard.open("a", encoding="utf-8") as f:
        f.write(line)


def read_records(results_dir: Path) -> list[Record]:
    """Read every `*.jsonl` shard in `results_dir`, return the union of records.

    Shards are read in sorted filename order for determinism. Order across
    shards is not otherwise meaningful; callers that need chronological order
    sort by each record's `timestamp` field.

    Returns [] if the directory does not exist. Skips blank lines. Raises
    ValueError on any non-blank line that is not valid JSON (corruption).
    """
    if not results_dir.exists():
        return []
    out: list[Record] = []
    for shard in sorted(results_dir.glob("*.jsonl")):
        for i, raw in enumerate(shard.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"results store corruption in {shard.name} at line {i}: {exc}"
                ) from exc
    return out
```

- [ ] **Step 4: Rename `results_store` → `results_dir` in `settings_loader.py`**

In the `Paths` dataclass, change `results_store: Path` to `results_dir: Path`.
In `load_settings`, change `results_store=_under_root(p["results_store"])` to `results_dir=_under_root(p["results_dir"])`.

- [ ] **Step 5: Update `settings.toml` and `conftest.py`**

In `factory/config/settings.toml`, change:

```toml
results_store    = "factory/data/results.json"
```

to:

```toml
results_dir      = "factory/data/results"
```

In `factory/tests/conftest.py`, change the `tmp_settings_file` fixture line:

```python
        results_store    = "factory/data/results.json"
```

to:

```python
        results_dir      = "factory/data/results"
```

- [ ] **Step 6: Update the 4 `write_record` call sites in `cycle.py`**

In `factory/cycle.py`, every call `write_record(paths.results_store, rec)` (lines 100, 137, 196, 262) becomes:

```python
        write_record(paths.results_dir, rec, node_id=s.node_id)
```

(Keep each line's existing indentation.)

- [ ] **Step 7: Update the 4 `read_records` call sites in `dashboard/server.py`**

In `factory/dashboard/server.py`, every `read_records(settings.paths.results_store)` (lines 80, 102, 107, 116) becomes `read_records(settings.paths.results_dir)`.

- [ ] **Step 8: Update `test_settings_loader.py`**

Change line 26 from `assert s.paths.results_store.is_relative_to(root)` to `assert s.paths.results_dir.is_relative_to(root)`.

- [ ] **Step 9: Update `test_dashboard.py`**

Replace the `_write_records` helper (lines 7–11) with a shard writer:

```python
def _write_records(results_dir: Path, recs: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    shard = results_dir / "local.jsonl"
    with shard.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
```

Change line 19 from `_write_records(s.paths.results_store, [` to `_write_records(s.paths.results_dir, [`.

- [ ] **Step 10: Update results-store reads in `test_cycle.py` and `test_integration_failures.py`**

In `factory/tests/test_cycle.py`, the two `read_records(s.paths.results_store)` calls (lines 48, 137) become `read_records(s.paths.results_dir)`.

In `factory/tests/test_integration_failures.py`, the three `read_records(s.paths.results_store)` calls (lines 66, 97, 159) become `read_records(s.paths.results_dir)`.

- [ ] **Step 11: Update the inline TOML in `test_integration_smoke.py`**

In `factory/tests/test_integration_smoke.py`, line 26, change:

```python
results_store    = "{(tmp_path / 'results.json').as_posix()}"
```

to:

```python
results_dir      = "{(tmp_path / 'results').as_posix()}"
```

- [ ] **Step 12: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS. (The `@pytest.mark.slow` smoke test may be deselected by default; that is fine.)

- [ ] **Step 13: Commit**

```bash
git add factory/results.py factory/settings_loader.py factory/config/settings.toml factory/cycle.py factory/dashboard/server.py factory/tests/
git commit -m "feat(factory): shard results store per machine (results_dir)"
```

---

## Task 5: Dedup log — per-machine shards

Replace the single `factory/data/dedup_log.txt` with a directory `factory/data/dedup/`. Each machine appends timestamped lines (`<unix-int>\t<summary>`) to its own shard `<node_id>.txt`; `read_tail` unions all shards, sorts by timestamp, and returns the globally most-recent `n` summaries oldest-first. Lines with no tab (pre-migration) are treated as timestamp 0.

**Files:**
- Modify: `factory/dedup.py` (`append_summary`, `read_tail`)
- Modify: `factory/settings_loader.py` (`Paths.dedup_log` → `dedup_dir`)
- Modify: `factory/config/settings.toml`
- Modify: `factory/tests/conftest.py`
- Modify: `factory/cycle.py:82,116`
- Test: `factory/tests/test_dedup.py`, `test_cycle.py`, `test_integration_failures.py`, `test_integration_smoke.py`

- [ ] **Step 1: Rewrite `factory/tests/test_dedup.py`**

Replace the entire file with:

```python
from pathlib import Path

from factory.dedup import append_summary, read_tail


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "first idea", node_id="local")
    append_summary(d, "second idea", node_id="local")
    append_summary(d, "third idea", node_id="local")
    assert read_tail(d, n=10) == ["first idea", "second idea", "third idea"]


def test_append_writes_timestamped_line(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "an idea", node_id="desk")
    raw = (d / "desk.txt").read_text(encoding="utf-8").strip()
    ts_str, sep, summary = raw.partition("\t")
    assert sep == "\t"
    assert ts_str.isdigit()
    assert summary == "an idea"


def test_read_tail_unions_shards_by_timestamp(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    (d / "desk.txt").write_text("100\tdesk old\n300\tdesk new\n", encoding="utf-8")
    (d / "laptop.txt").write_text("200\tlaptop mid\n", encoding="utf-8")
    # Globally sorted by timestamp, oldest first.
    assert read_tail(d, n=10) == ["desk old", "laptop mid", "desk new"]


def test_read_tail_caps_at_n(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    lines = "".join(f"{i}\tidea {i}\n" for i in range(50))
    (d / "local.txt").write_text(lines, encoding="utf-8")
    tail = read_tail(d, n=30)
    assert len(tail) == 30
    assert tail[0] == "idea 20"
    assert tail[-1] == "idea 49"


def test_read_tail_legacy_untimestamped_line_is_oldest(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    d.mkdir()
    # A pre-migration line has no tab -> treated as timestamp 0 (always oldest).
    (d / "local.txt").write_text("legacy idea no tab\n500\tnew idea\n", encoding="utf-8")
    assert read_tail(d, n=10) == ["legacy idea no tab", "new idea"]


def test_read_tail_handles_missing_dir(tmp_path: Path) -> None:
    assert read_tail(tmp_path / "does_not_exist", n=30) == []


def test_append_strips_newlines_inside_summary(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "line1\nline2\rline3", node_id="local")
    assert read_tail(d, n=10) == ["line1 line2 line3"]


def test_append_skips_empty_or_whitespace(tmp_path: Path) -> None:
    d = tmp_path / "dedup"
    append_summary(d, "", node_id="local")
    append_summary(d, "   ", node_id="local")
    append_summary(d, "real entry", node_id="local")
    assert read_tail(d, n=10) == ["real entry"]
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `python -m pytest factory/tests/test_dedup.py -q`
Expected: FAIL — `append_summary` does not accept `node_id`.

- [ ] **Step 3: Rewrite `factory/dedup.py`**

Replace the entire file with:

```python
from __future__ import annotations

import time
from pathlib import Path


def append_summary(dedup_dir: Path, summary: str, *, node_id: str) -> None:
    """Append one timestamped one_line_summary to this machine's dedup shard.

    The shard is `dedup_dir/<node_id>.txt`; each line is `<unix-int>\\t<summary>`.
    Newlines/carriage returns inside the summary are replaced with spaces so
    one line == one entry. Empty/whitespace-only summaries are silently ignored.
    Parent directories are created on demand.
    """
    cleaned = " ".join(summary.replace("\r", "\n").split("\n")).strip()
    if not cleaned:
        return
    shard = dedup_dir / f"{node_id}.txt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    with shard.open("a", encoding="utf-8") as f:
        f.write(f"{ts}\t{cleaned}\n")


def read_tail(dedup_dir: Path, n: int) -> list[str]:
    """Return the globally most-recent `n` summaries across all shards, oldest first.

    Reads every `*.txt` shard in `dedup_dir`, parses each line as
    `<timestamp>\\t<summary>`, sorts all entries by timestamp ascending, and
    returns the summaries of the last `n`.

    Legacy tolerance: a line with no tab (a pre-migration entry) is treated as
    timestamp 0, i.e. always oldest. Returns [] if the directory does not exist
    or if n <= 0.
    """
    if n <= 0 or not dedup_dir.exists():
        return []
    entries: list[tuple[int, str]] = []
    for shard in sorted(dedup_dir.glob("*.txt")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ts_str, sep, rest = line.partition("\t")
            if sep:
                try:
                    ts = int(ts_str)
                    summary = rest
                except ValueError:
                    ts, summary = 0, line
            else:
                ts, summary = 0, line
            entries.append((ts, summary))
    entries.sort(key=lambda e: e[0])
    return [summary for _, summary in entries[-n:]]
```

- [ ] **Step 4: Rename `dedup_log` → `dedup_dir` in `settings_loader.py`**

In the `Paths` dataclass, change `dedup_log: Path` to `dedup_dir: Path`.
In `load_settings`, change `dedup_log=_under_root(p["dedup_log"])` to `dedup_dir=_under_root(p["dedup_dir"])`.

- [ ] **Step 5: Update `settings.toml` and `conftest.py`**

In `factory/config/settings.toml`, change:

```toml
dedup_log        = "factory/data/dedup_log.txt"
```

to:

```toml
dedup_dir        = "factory/data/dedup"
```

In `factory/tests/conftest.py`, change the fixture line:

```python
        dedup_log        = "factory/data/dedup_log.txt"
```

to:

```python
        dedup_dir        = "factory/data/dedup"
```

- [ ] **Step 6: Update the `cycle.py` dedup call sites**

In `factory/cycle.py`, line 82:

```python
    dedup_tail = read_tail(paths.dedup_log, n=30)
```

becomes:

```python
    dedup_tail = read_tail(paths.dedup_dir, n=30)
```

Line 116:

```python
    append_summary(paths.dedup_log, parsed["one_line_summary"])
```

becomes:

```python
    append_summary(paths.dedup_dir, parsed["one_line_summary"], node_id=s.node_id)
```

- [ ] **Step 7: Update dedup reads in `test_cycle.py`**

In `factory/tests/test_cycle.py`:

Line 45 — replace:

```python
    assert not s.paths.dedup_log.exists() or s.paths.dedup_log.read_text().strip() == ""
```

with:

```python
    _dedup_shard = s.paths.dedup_dir / "local.txt"
    assert not _dedup_shard.exists() or _dedup_shard.read_text(encoding="utf-8").strip() == ""
```

Line 69 — `read_tail(s.paths.dedup_log, n=10)` becomes `read_tail(s.paths.dedup_dir, n=10)`.
Line 177 — `read_tail(s.paths.dedup_log, n=10)` becomes `read_tail(s.paths.dedup_dir, n=10)`.

- [ ] **Step 8: Update dedup reads in `test_integration_failures.py`**

In `factory/tests/test_integration_failures.py`:

Line 71 — replace:

```python
    assert not s.paths.dedup_log.exists() or s.paths.dedup_log.read_text().strip() == ""
```

with:

```python
    _dedup_shard = s.paths.dedup_dir / "local.txt"
    assert not _dedup_shard.exists() or _dedup_shard.read_text(encoding="utf-8").strip() == ""
```

Line 100 — `read_tail(s.paths.dedup_log, n=10)` becomes `read_tail(s.paths.dedup_dir, n=10)`.
Line 164 — `read_tail(s.paths.dedup_log, n=10)` becomes `read_tail(s.paths.dedup_dir, n=10)`.

- [ ] **Step 9: Update the inline TOML in `test_integration_smoke.py`**

In `factory/tests/test_integration_smoke.py`, line 25, change:

```python
dedup_log        = "{(tmp_path / 'dedup.txt').as_posix()}"
```

to:

```python
dedup_dir        = "{(tmp_path / 'dedup').as_posix()}"
```

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add factory/dedup.py factory/settings_loader.py factory/config/settings.toml factory/cycle.py factory/tests/
git commit -m "feat(factory): shard dedup log per machine with timestamps (dedup_dir)"
```

---

## Task 6: Registry auto-discovery

Stop editing `backtester/strategies/registry.py` per strategy. Instead, the registry auto-discovers `strategies/gen_*.py` at import time. `append_registry_entry` and `RegistryAlreadyHasStrategy` are removed. Curated (hand-written) strategies keep their explicit registrations.

**Files:**
- Modify: `backtester/strategies/registry.py`
- Modify: `factory/filesystem.py` (remove `append_registry_entry`, `RegistryAlreadyHasStrategy`)
- Modify: `factory/cycle.py` (drop the registry-append import + call)
- Test: `tests/unit/test_strategy_registry.py`, `factory/tests/test_filesystem.py`, `factory/tests/test_cycle.py`, `factory/tests/test_integration_failures.py`, `factory/tests/test_integration_smoke.py`

- [ ] **Step 1: Write the failing auto-discovery tests**

Append to `tests/unit/test_strategy_registry.py`:

```python
from pathlib import Path
from unittest import mock


def test_curated_strategies_registered_on_import() -> None:
    """Importing the registry registers the curated hand-written strategies."""
    import backtester.strategies.registry as reg
    assert reg.get_strategy_class("sma_cross") is not None
    assert reg.get_strategy_class("mean_reversion_atr") is not None


def test_discover_only_imports_gen_modules_in_sorted_order() -> None:
    """discover_generated_strategies imports only strategies.gen_* modules,
    in sorted (deterministic) order."""
    from backtester.strategies.registry import discover_generated_strategies
    imported = discover_generated_strategies()
    assert all(name.startswith("strategies.gen_") for name in imported)
    assert imported == sorted(imported)


def test_discover_invokes_invalidate_caches() -> None:
    """importlib.invalidate_caches() is called before globbing so a
    just-written generated file is visible."""
    import backtester.strategies.registry as reg
    with mock.patch.object(reg.importlib, "invalidate_caches") as inv:
        reg.discover_generated_strategies()
    inv.assert_called_once()


def test_discover_skips_broken_generated_module(caplog) -> None:
    """A gen_*.py that fails to import is skipped (logged with its filename
    and exception), never fatal; valid modules still register."""
    import strategies as strategies_pkg
    from backtester.strategies.registry import discover_generated_strategies
    pkg_dir = Path(strategies_pkg.__file__).resolve().parent
    broken = pkg_dir / "gen_zzz_brokenfixture.py"
    broken.write_text("this is not valid python !!!\n", encoding="utf-8")
    try:
        with caplog.at_level("WARNING"):
            imported = discover_generated_strategies()   # must not raise
    finally:
        broken.unlink(missing_ok=True)
    assert "strategies.gen_zzz_brokenfixture" not in imported
    assert "gen_zzz_brokenfixture.py" in caplog.text
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `python -m pytest tests/unit/test_strategy_registry.py -q`
Expected: FAIL — `discover_generated_strategies` does not exist.

- [ ] **Step 3: Rewrite `backtester/strategies/registry.py`**

Replace the entire file with:

```python
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Dict, Type

from backtester.strategies.base import BaseStrategy

log = logging.getLogger(__name__)

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Register a strategy class by its `strategy_id`."""
    if not getattr(cls, "strategy_id", None):
        raise ValueError(f"{cls.__name__} is missing a non-empty `strategy_id`")
    STRATEGY_REGISTRY[cls.strategy_id] = cls
    return cls


def get_strategy_class(strategy_id: str) -> Type[BaseStrategy]:
    if strategy_id not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy {strategy_id!r} is not registered. "
            f"Known: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy_id]


def discover_generated_strategies() -> list[str]:
    """Import every `strategies/gen_*.py` and register its GeneratedStrategy.

    The factory's distributed design (no per-strategy registry edit) relies on
    this: every machine's generated strategies are picked up automatically.

    - `importlib.invalidate_caches()` is called first so a strategy file
      written earlier in the same process is visible.
    - Only `gen_*.py` is globbed; curated strategies are never touched here.
    - Filenames are sorted before import so registration order is the same
      on every machine and every test run.
    - Each import is wrapped: one broken generated module is skipped (its
      filename and full exception are logged), never aborting the import.

    Returns the list of module names imported, in import order.
    """
    import strategies as _strategies_pkg
    pkg_dir = Path(_strategies_pkg.__file__).resolve().parent
    importlib.invalidate_caches()
    imported: list[str] = []
    for path in sorted(pkg_dir.glob("gen_*.py")):
        module_name = f"strategies.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            register_strategy(module.GeneratedStrategy)
            imported.append(module_name)
        except Exception as exc:
            log.warning(
                "registry auto-discovery: skipping %s — %r",
                path.name, exc, exc_info=True,
            )
    return imported


# --- Curated strategy registrations (explicit, predictable order) ---
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402
from strategies.rsi_long_short import RSILongShortStrategy  # noqa: E402
from strategies.momentum_streak import MomentumStreakStrategy  # noqa: E402
from strategies.mean_reversion_atr import MeanReversionAtrStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
register_strategy(RSILongShortStrategy)
register_strategy(MomentumStreakStrategy)
register_strategy(MeanReversionAtrStrategy)

# --- Generated strategies: auto-discovered from strategies/gen_*.py ---
discover_generated_strategies()
```

(Note: `gen_1715800000` was previously in the curated block — it is a generated strategy and is now picked up by `discover_generated_strategies()` like every other `gen_*.py`.)

- [ ] **Step 4: Run the registry tests**

Run: `python -m pytest tests/unit/test_strategy_registry.py -q`
Expected: PASS — all tests including the four new ones.

- [ ] **Step 5: Remove `append_registry_entry` and `RegistryAlreadyHasStrategy` from `factory/filesystem.py`**

Delete the `RegistryAlreadyHasStrategy` class (lines 13–14) and the entire `append_registry_entry` function (lines 54–76). The file keeps `FilesystemError`, `pick_unused_strategy_id`, and `write_strategy_artifacts`. After the edit the top of the file is:

```python
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class FilesystemError(RuntimeError):
    pass


def pick_unused_strategy_id(base: str, *, strategies_dir: Path) -> str:
```

- [ ] **Step 6: Drop the registry-append from `factory/cycle.py`**

In `factory/cycle.py`, change the `factory.filesystem` import block (lines 12–17) from:

```python
from factory.filesystem import (
    RegistryAlreadyHasStrategy,
    append_registry_entry,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)
```

to:

```python
from factory.filesystem import (
    pick_unused_strategy_id,
    write_strategy_artifacts,
)
```

Delete the registry-append call (lines 150–153):

```python
    try:
        append_registry_entry(strategy_id=strategy_id, registry_file=paths.registry_file)
    except RegistryAlreadyHasStrategy:
        log.info("registry already has %s; continuing", strategy_id)
```

The `write_strategy_artifacts(...)` call immediately above it stays. The comment on line 142 (`# Step 8-10: write files + register.`) should become `# Step 8-9: write files. Registry is auto-discovery now (no per-strategy edit).`

- [ ] **Step 7: Remove the `append_registry_entry` tests from `test_filesystem.py`**

In `factory/tests/test_filesystem.py`, change the import block to:

```python
from factory.filesystem import (
    FilesystemError,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)
```

Delete the `_seed_registry` helper (lines 14–23) and the four registry tests: `test_append_registry_entry_adds_two_lines`, `test_append_registry_is_idempotent`, `test_append_registry_does_not_false_positive_on_prefix_match` (lines 53–83). Keep the `write_strategy_artifacts` and `pick_unused_strategy_id` tests.

- [ ] **Step 8: Remove obsolete registry assertions in `test_cycle.py`**

The cycle no longer edits the registry, so these assertions are obsolete:

In `test_complete_cycle_writes_files_registry_record`, delete:

```python
    # Registry has the entry.
    assert "_gen_local_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")
```

In `test_stage_failure_writes_failed_record_keeps_dedup_and_files`, change:

```python
    # Files + registry ARE present (per §9 landmine 2: orphan accepted).
    assert (s.paths.strategies_dir / "gen_local_1715800000.py").exists()
    assert "_gen_local_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")
```

to:

```python
    # The strategy file IS present (orphan accepted); the registry is no
    # longer edited per-strategy — it auto-discovers gen_*.py at import.
    assert (s.paths.strategies_dir / "gen_local_1715800000.py").exists()
```

In `test_validation_failure_writes_dedup_but_no_files`, delete:

```python
    # Registry is untouched.
    reg_text = s.paths.registry_file.read_text(encoding="utf-8")
    assert "gen_local_1715800000" not in reg_text
```

- [ ] **Step 9: Remove the obsolete registry assertion in `test_integration_failures.py`**

In `test_stage_failure_records_correct_failed_stage`, change:

```python
    # Files + registry kept (§9 landmine 2: orphan is accepted).
    assert (s.paths.strategies_dir / f"{strategy_id}.py").exists()
    assert f"_{strategy_id}" in s.paths.registry_file.read_text(encoding="utf-8")
```

to:

```python
    # The strategy file is kept (orphan accepted). The registry is not
    # edited per-strategy any more (auto-discovery replaces it).
    assert (s.paths.strategies_dir / f"{strategy_id}.py").exists()
```

The `test_validation_failure_keeps_dedup_no_files` assertion `assert "gen_xx" not in s.paths.registry_file.read_text(...)` stays valid (the registry is never touched) — leave it.

- [ ] **Step 10: Simplify the registry cleanup in `test_integration_smoke.py`**

The smoke test's `finally` block strips smoke-id lines out of the real `registry.py`. The cycle no longer writes to the registry, so that step is dead. In `factory/tests/test_integration_smoke.py`, replace the `finally` block (lines 86–100) with:

```python
    finally:
        # Cleanup: remove the smoke strategy + config so the backtester repo
        # isn't permanently polluted. The registry is auto-discovery now and
        # is never edited, so there is nothing to strip there.
        strat_file = repo / "strategies" / f"{smoke_id}.py"
        cfg_file = repo / "configs" / "wfo" / f"{smoke_id}.yaml"
        if strat_file.exists():
            strat_file.unlink()
        if cfg_file.exists():
            cfg_file.unlink()
```

- [ ] **Step 11: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add backtester/strategies/registry.py factory/filesystem.py factory/cycle.py factory/tests/ tests/unit/test_strategy_registry.py
git commit -m "feat: auto-discover generated strategies; drop registry edit"
```

---

## Task 7: `factory/sync.py` — git coordination module

Create the module that coordinates the shared pool through git: `bootstrap` (one-time branch setup + legacy fold), `sync_pull` (before a cycle), `sync_push` (after a cycle). All three are no-ops when `[sync] enabled = false`. Tested entirely offline against throwaway local git repos.

**Files:**
- Create: `factory/sync.py`
- Test: `factory/tests/test_sync.py`

- [ ] **Step 1: Write `factory/tests/test_sync.py`**

Create `factory/tests/test_sync.py`:

```python
"""Offline tests for factory/sync.py — no network, no real GitHub.

Each test scaffolds throwaway local git repos: a bare repo acts as the
"remote", and one or two clones act as factory machines (nodes).
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

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


def test_sync_push_noop_when_nothing_changed(tmp_path: Path, caplog) -> None:
    repo = _clone(_init_bare_remote(tmp_path / "remote.git"), tmp_path / "node")
    _seed_master(repo)
    s = _node_settings(repo, "desk")
    bootstrap(s)
    sync_pull(s)
    with caplog.at_level("INFO"):
        sync_push(s)   # nothing produced this cycle
    assert "no-op" in caplog.text.lower()


def test_sync_pull_skips_on_dirty_tree(tmp_path: Path, caplog) -> None:
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
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `python -m pytest factory/tests/test_sync.py -q`
Expected: FAIL — `factory.sync` does not exist (ImportError).

- [ ] **Step 3: Create `factory/sync.py`**

Create `factory/sync.py`:

```python
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
    result/dedup shards. Only existing paths are returned (git add of a
    missing pathspec errors)."""
    p = settings.paths
    dirs = [p.strategies_dir, p.configs_dir, p.results_dir, p.dedup_dir]
    return [str(d) for d in dirs if d.exists()]


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
        _git(["pull", "--rebase", remote, branch], cwd=root)
    raise SyncError(
        f"sync_push: push still failing after {settings.sync.push_retries} retries"
    )
```

- [ ] **Step 4: Run the sync tests**

Run: `python -m pytest factory/tests/test_sync.py -q`
Expected: PASS — all nine tests.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add factory/sync.py factory/tests/test_sync.py
git commit -m "feat(factory): add git sync module (bootstrap/pull/push)"
```

---

## Task 8: Loop integration — wire sync around each cycle

`run_loop` calls `bootstrap()` once at start, then `sync_pull()` before each `run_cycle()` and `sync_push()` after. Each sync call is wrapped so a failure is logged and the loop continues. With `[sync] enabled = false` every wrapper hits a no-op, so the existing loop tests are unaffected.

**Files:**
- Modify: `factory/loop.py`
- Test: `factory/tests/test_sync.py` (add a loop-integration test)

- [ ] **Step 1: Write the failing test**

Append to `factory/tests/test_sync.py`:

```python
def test_run_loop_swallows_sync_failure(tmp_path: Path) -> None:
    """With sync enabled but the remote unreachable, run_loop logs the sync
    failures and still completes its cycles — sync failure never aborts the
    loop."""
    import random
    from unittest import mock
    from factory.cycle import CycleOutcome
    from factory.loop import run_loop

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
    with mock.patch("factory.loop.run_cycle", return_value=fake) as rc:
        completed = run_loop(s, rng=random.Random(0), max_cycles_override=1)
    assert rc.call_count == 1
    assert completed == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest factory/tests/test_sync.py::test_run_loop_swallows_sync_failure -q`
Expected: FAIL — `run_loop` does not call sync; a `SyncError` may escape, or `rc.call_count`/`completed` is wrong because sync is never attempted. (The test asserts the wired behavior.)

- [ ] **Step 3: Import the sync functions in `loop.py`**

In `factory/loop.py`, after `from factory.settings_loader import Settings, load_settings` add:

```python
from factory.sync import bootstrap, sync_pull, sync_push
```

- [ ] **Step 4: Call `bootstrap` at the start of `run_loop`**

In `run_loop`, immediately after `sleep_sec = settings.loop.inter_cycle_sleep_sec` and before `completed = 0`, add:

```python
    try:
        bootstrap(settings)
    except Exception as exc:
        log.exception("sync bootstrap failed (continuing): %s", exc)
```

- [ ] **Step 5: Wrap each cycle with `sync_pull` / `sync_push`**

In `run_loop`, the body of the `while not flag.is_set():` loop currently starts with `try: outcome = run_cycle(...)`. Change the loop body so it reads:

```python
    while not flag.is_set():
        try:
            sync_pull(settings)
        except Exception as exc:
            log.exception("sync_pull failed (continuing): %s", exc)
        try:
            outcome = run_cycle(settings, rng=rng)
            log.info("cycle %d outcome=%s id=%s",
                     completed + 1, outcome.status, outcome.strategy_id)
        except Exception as exc:
            # An unexpected exception from inside run_cycle: log and continue.
            # (run_cycle is supposed to never raise on expected failures, so
            # reaching here means a bug — but the loop must not die.)
            log.exception("unexpected exception in run_cycle: %s", exc)
        try:
            sync_push(settings)
        except Exception as exc:
            log.exception("sync_push failed (continuing): %s", exc)
        completed += 1
        if max_cycles and completed >= max_cycles:
            break
        if flag.is_set():
            break
        if sleep_sec > 0:
            # Sleep in short increments so SIGINT is responsive.
            slept = 0.0
            while slept < sleep_sec and not flag.is_set():
                time.sleep(min(0.5, sleep_sec - slept))
                slept += 0.5
```

- [ ] **Step 6: Run the sync + loop tests**

Run: `python -m pytest factory/tests/test_sync.py factory/tests/test_loop.py -q`
Expected: PASS — the new loop test, plus the existing `test_loop.py` tests (sync disabled → no-ops).

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest factory/tests/ tests/unit/test_strategy_registry.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add factory/loop.py factory/tests/test_sync.py
git commit -m "feat(factory): run sync_pull/sync_push around each cycle"
```

---

## Task 9: Fix the `endurance_check.py` operator script

`factory/scripts/endurance_check.py` is an operator script (not a pytest test). It writes an inline `settings.toml` with the now-removed `dedup_log` / `results_store` keys, reads results via the old API, and asserts on registry text that the factory no longer writes. Bring it in line with the new stores and the auto-discovery registry.

**Files:**
- Modify: `factory/scripts/endurance_check.py`

- [ ] **Step 1: Update the inline `settings.toml` keys**

In `_load_settings`, change the two `[paths]` lines:

```python
dedup_log        = "{(scratch / 'dedup.txt').as_posix()}"
results_store    = "{(scratch / 'results.json').as_posix()}"
```

to:

```python
dedup_dir        = "{(scratch / 'dedup').as_posix()}"
results_dir      = "{(scratch / 'results').as_posix()}"
```

- [ ] **Step 2: Update the results read**

In `main`, change:

```python
    records = read_records(s.paths.results_store)
```

to:

```python
    records = read_records(s.paths.results_dir)
```

- [ ] **Step 3: Replace the obsolete registry-duplicate check**

The factory no longer appends to `registry.py`, so the registry-text duplicate check (the block from `# Registry must have NO duplicate gen_endurance_*` through the `assert not duplicates` and the `unique_ids` print, roughly lines 150–163) is obsolete. Replace that whole block with a strategy-shard sanity check:

```python
    # Every cycle that locked in a strategy id appends exactly one record to
    # this machine's shard. The shard count must equal the records that have
    # a non-null strategy_id.
    with_ids = sum(1 for r in records if r.get("strategy_id"))
    print(f"records with a strategy_id: {with_ids}")
```

(Keep everything else in `main` — the record-count assertion, the status tally, and the final `print("ENDURANCE CHECK PASSED")`.)

- [ ] **Step 4: Verify the script imports cleanly**

Run: `python -c "import factory.scripts.endurance_check"`
Expected: no error (the module imports without executing `main`).

- [ ] **Step 5: Smoke-run a short endurance check**

Run: `python -m factory.scripts.endurance_check --cycles 3`
Expected: prints `ENDURANCE CHECK PASSED` (this runs 3 real backtester subprocess cycles; it takes a minute or two).

- [ ] **Step 6: Commit**

```bash
git add factory/scripts/endurance_check.py
git commit -m "chore(factory): update endurance_check for sharded stores"
```

---

## Self-Review

**Spec coverage** — each spec section maps to a task:

| Spec § | Requirement | Task |
|---|---|---|
| §2 | `node_id` setting, validated, default `local`, fatal on malformed | Task 2 |
| §3 | `gen_<node_id>_<ts>` IDs; `_2/_3` fallback kept | Task 3 (fallback in `pick_unused_strategy_id` is untouched) |
| §4 | Results → per-machine `*.jsonl` shards, union read | Task 4 |
| §5 | Dedup → timestamped `*.txt` shards, union `read_tail`, legacy tolerance | Task 5 |
| §6 | Registry auto-discovery; `append_registry_entry` removed | Task 6 |
| §7 | `factory/sync.py` — `bootstrap`/`sync_pull`/`sync_push` | Task 7 |
| §8 | Loop integration; no-op when disabled | Task 8 |
| §9 | Pool-wide reads (dashboard) | Task 4, Step 7 |
| §11 | All test categories | Tasks 2–8 tests |
| §12 | File layout | Tasks 1–9 cover every listed file |
| §14 | Acceptance criteria 1–8 | Criteria 1→T2, 2→T3, 3→T4, 4→T5, 5→T6, 6→T7, 7→T2/T7, 8→all |

**Two spec inaccuracies, deliberately not implemented:**
- §9 / §12 claim `factory/promote.py` reads results and needs a `results_dir` change. It does not — `factory/promote.py` contains only `promote_strategy` and its helpers; it never imports or calls `read_records`. No change to `promote.py` is in this plan, and that is correct.
- §9 says the dashboard "shows the whole pool"; the dashboard reverses `read_records` output for newest-first display. After Task 4 that reversal operates on sorted-shard-concatenation, not a true global timestamp order — but §9 explicitly mandates a one-line, no-logic-change update for the dashboard, so the plan keeps it that way. Pool-wide spend (`_aggregate`'s `cumulative_spend`) already sums across all records once `read_records` unions shards — no extra work needed.

**`registry_file` setting:** kept in `Paths` and `settings.toml`. The cycle no longer uses it (Task 6), but the spec does not ask for its removal and `test_settings_loader.py` still asserts it; removing it would be unrequested churn.

**Type consistency:** `write_record(results_dir, record, *, node_id)` and `append_summary(dedup_dir, summary, *, node_id)` use keyword-only `node_id` consistently across the module definition (Tasks 4, 5) and every call site in `cycle.py`. `read_records(results_dir)` / `read_tail(dedup_dir, n)` take a directory everywhere. `discover_generated_strategies()` is referenced only in `registry.py` and `test_strategy_registry.py`, same name. `SyncError`, `bootstrap`, `sync_pull`, `sync_push`, `_fold_legacy_stores` are defined in `factory/sync.py` and imported with matching names in `loop.py` and `test_sync.py`.

**Suite-green invariant:** Task 1 is additive. Task 2 is additive (no field removed). Tasks 3–8 each rename/replace and fix every call site within the same commit. Task 9 fixes a script not collected by pytest (the suite was already green without it; it was knowingly left broken from Task 4 until here).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-distributed-factory.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
