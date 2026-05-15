# Distributed Strategy Factory — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Let multiple geographically-separate machines, each running the strategy factory, contribute generated strategies into one shared pool — coordinated entirely through the existing GitHub repository, with no new servers, services, or databases.

**Motivation:** Throughput. The factory cycle is slow (an LLM generation call plus backtest/optimize/WFO stages, easily over an hour per strategy). Running the factory on N of the user's own machines grows the strategy pool roughly N× faster.

**Approach in one sentence:** Every machine writes only files that it alone owns; the shared pool is the union of all machines' files on a dedicated `factory-pool` git branch; coordination is `git pull --rebase` before a cycle and `git push` (with bounded retry) after.

**Context — the single-machine assumptions being removed:** The factory today is a single-process, single-machine loop. Each cycle appends to one `factory/data/results.json` and one `factory/data/dedup_log.txt`, does a read-modify-write on `backtester/strategies/registry.py`, and mints strategy IDs as `gen_<unix-second>`. With no locking, two machines sharing storage would corrupt all three files and collide IDs. This spec removes every shared-mutable-file touchpoint.

---

## 1. Core principle

**No two machines ever write the same file.** If that holds, `git pull --rebase` is always conflict-free and `git push` always converges. Everything machine-owned is keyed by a per-machine `node_id`. The factory pool is the set-union of every machine's owned files on the `factory-pool` branch; "reading the pool" means reading all shards.

Three classes of shared state are converted:
- **Append-style stores** (results, dedup) → per-machine shards, union on read.
- **Create-once files** (strategy `.py`, config `.yaml`) → globally-unique names via `node_id` so they never collide.
- **Read-modify-write file** (`registry.py`) → eliminated by auto-discovery.

---

## 2. Node identity

Each machine has a stable short `node_id` — e.g. `desk`, `laptop`, `vps1`. It is set per-machine in `settings.local.toml` (already gitignored, already per-machine, so this adds nothing to version control).

- New settings key: `node_id` (string). Read once at startup.
- Constraint: must match `^[a-z0-9][a-z0-9-]*$` (used in filenames and git-safe paths). Validated at settings load; a missing or malformed `node_id` is a fatal startup error with a clear message.
- Single-machine default: if the user never sets one, default `node_id = "local"`. The factory then behaves as today (one shard named `local`).

---

## 3. Strategy IDs

Today: `gen_<unix-second>` (`factory/cycle.py`, `gen_{_now_unix_int()}`). Collision-prone across machines and within a machine in the same second.

New: `gen_<node_id>_<unix-second>`. Example: `gen_desk_1778829071`.

- The existing `pick_unused_strategy_id` `_2/_3` suffix is kept as the same-machine, same-second fallback.
- Because `node_id` is in the ID, the derived create-once files — `strategies/gen_desk_1778829071.py` and `configs/wfo/gen_desk_1778829071.yaml` — are globally unique. Different machines' new files merge cleanly (distinct filenames); no machine overwrites another's.

---

## 4. Results store → per-machine shards

Today: a single file `factory/data/results.json` (JSONL), appended by `results.write_record`, read whole by `results.read_records`.

New:
- A directory `factory/data/results/`. Each machine appends only to its own shard `factory/data/results/<node_id>.jsonl`.
- `write_record` appends to the current machine's shard (path derived from `node_id`).
- `read_records` reads and concatenates every `*.jsonl` file in the directory, returning all records. Order across shards is not guaranteed; callers that need ordering sort by the record's `timestamp` field (already present on every record).
- Settings: replace the `results_store` file key with `results_dir = "factory/data/results"`.

Because each shard has exactly one writer (one machine), shards never conflict on push/pull.

---

## 5. Dedup log → per-machine shards

Today: a single file `factory/data/dedup_log.txt`, one `one_line_summary` per line; `dedup.append_summary` appends, `dedup.read_tail(n)` returns the last `n` lines (oldest-first within the tail).

New:
- A directory `factory/data/dedup/`. Each machine appends only to `factory/data/dedup/<node_id>.txt`.
- **Each line is timestamped:** `<unix-int>\t<summary>`. The summary still has internal newlines/CRs replaced by spaces (unchanged from today).
- `append_summary` writes a timestamped line to the current machine's shard. The deliberate early-capture behavior is preserved per shard: the append still happens at cycle step 6 (right after idea extraction, before validation), so a mid-cycle crash still records the idea.
- `read_tail(n)` reads every `*.txt` shard in the directory, parses `(timestamp, summary)` pairs, sorts by timestamp ascending, and returns the summaries of the most recent `n` — oldest-first, matching what `build_prompt` expects today. The prompt's "last 30 ideas" therefore spans all machines.
- **Legacy tolerance:** a line with no tab (a pre-migration entry) is treated as timestamp `0`, i.e. always oldest. This lets migration copy old content verbatim.
- Settings: replace the `dedup_log` file key with `dedup_dir = "factory/data/dedup"`.

---

## 6. Registry → auto-discovery for generated strategies

Today: `factory/filesystem.py:append_registry_entry` does a read-modify-write on `backtester/strategies/registry.py`, adding an `import` + `register_strategy(...)` line per generated strategy. Across machines this conflicts on nearly every cycle.

New:
- **`append_registry_entry` is removed.** Cycle step 8-10 no longer edits `registry.py`.
- `backtester/strategies/registry.py` keeps its explicit registrations for the curated, hand-written strategies (`sma_cross`, `mean_reversion_atr`, …) — **unchanged**.
- It additionally runs an **auto-discovery** pass at import time that registers every generated strategy. Discovery:
  1. Calls `importlib.invalidate_caches()` first — the factory's functional-validation stage can import a just-written strategy in the same process, and Python's import machinery can otherwise miss a newly created file.
  2. Globs `strategies/gen_*.py` only — curated strategies are never touched by discovery.
  3. **Sorts the discovered filenames** before importing, so registration order is deterministic across machines and test runs.
  4. Imports each module and calls `register_strategy` on its `GeneratedStrategy` class.
  5. Wraps each per-file import in `try/except`: a single broken generated module is skipped, never aborting the whole registry import. The skip **must log the offending filename and the full exception** (not a bare "skipped" message) — in a large pool, a silently-absent generated strategy is otherwise painful to diagnose. (Generated files are written only after static + functional validation, so this is a safety net, not the common path.)

This is a **mechanism swap, not a behavioral change**: the factory already registers every validated generated strategy today, and importing `registry.py` already imports all of them. After this change there is no per-strategy file edit, so the last cross-machine conflict point is gone. It is also the only change that touches `backtester/strategies/`.

---

## 7. Git sync — `factory/sync.py` (new module)

A new module wrapping `git` (via `subprocess`). Three operations:

- **`bootstrap()`** — one-time, idempotent. Ensures the `factory-pool` branch exists: if absent, it is created off `master` locally **and pushed to the remote**. Publishing the branch to the remote on first run is **intentional remote-mutating behavior** — it is the one bootstrap action that touches remote state, and it is deliberate, because the distributed pool cannot function until the branch is visible to the other machines. `bootstrap()` also ensures local-scratch directories are gitignored (`factory/data/_tmp/`, `factory/logs/`, `output/runs/`), and folds any pre-existing single-file stores into this machine's shards: `factory/data/results.json` → `results/<node_id>.jsonl`, `factory/data/dedup_log.txt` → `dedup/<node_id>.txt` (verbatim copy; legacy un-timestamped dedup lines are tolerated per §5). Safe to run repeatedly.
- **`sync_pull()`** — called at cycle start. Ensures the working tree is on `factory-pool`, then `git fetch` + `git pull --rebase`. **Dirty-tree handling:** before rebasing, `sync_pull` checks `git status --porcelain` for tracked changes. Gitignored local scratch (`_tmp/`, `logs/`, `output/runs/`) never appears there, so a tree where the previous `sync_push` committed cleanly is clean. If unexpected *tracked* dirt is present, `sync_pull` logs a warning and **skips the sync for this cycle** rather than attempting a rebase that git would reject — generation still proceeds, and the next clean cycle catches up. Brings in other machines' strategies, configs, and result/dedup shards.
- **`sync_push()`** — called at cycle end. `git add` this cycle's new strategy file, config file, and this machine's two shards. **No-op when nothing changed:** after staging, `sync_push` checks `git diff --cached --quiet`; if there is no staged diff — e.g. a cycle that produced no new tracked files — it skips the commit and push entirely and logs a no-op. Otherwise it commits and pushes. On a non-fast-forward rejection (another machine pushed first), `git pull --rebase` and retry, bounded by `push_retries` (default 5). Because every machine writes only its own shards and uniquely-named files, the rebase is always conflict-free, so retries converge immediately.

**Settings — new `[sync]` section:**
- `enabled` (bool, default `false`) — master switch.
- `branch` (default `"factory-pool"`).
- `remote` (default `"origin"`).
- `push_retries` (int, default `5`).

**Failure handling:** any sync failure (network down, auth expired, an unexpected conflict) is logged; the cycle and loop continue. A machine that cannot reach GitHub keeps generating locally and catches up on the next successful sync. **Sync failure never aborts generation.**

---

## 8. Loop & cycle integration

- `factory/loop.py:run_loop` calls `sync_pull()` immediately before each `run_cycle()` and `sync_push()` immediately after — each wrapped so a failure logs and the loop proceeds.
- `node_id` is read once from settings and threaded into ID minting (`factory/cycle.py`) and shard-path resolution (`results.py`, `dedup.py`).
- `run_cycle`'s stage order is otherwise unchanged.
- When `[sync] enabled = false`, `sync_pull`/`sync_push`/`bootstrap` are no-ops; the factory runs exactly as the single-machine version, just with the new ID scheme and sharded paths (both harmless on one machine).

---

## 9. Reads become pool-wide

`read_records` (§4) and `read_tail` (§5) become shard-union readers. They still take a path argument, but it is now the **directory** (`results_dir` / `dedup_dir`) rather than a single file. Downstream consumers therefore need only a one-line call-site change — the settings key they pass — and **no logic change**:
- `factory/dashboard/server.py` — passes `results_dir` to `read_records`; shows the whole pool after a pull. "The pool" is the union of shards on `factory-pool`.
- `factory/promote.py` — passes `results_dir`; reads results pool-wide for promotion decisions.
- The dashboard may additionally sum `total_cost_usd` across all records to display pool-wide LLM spend (optional, low-effort).

---

## 10. Consequences (stated, not chosen)

- **Cost scales linearly.** N machines = roughly N× the LLM spend. Every cycle already records `total_cost_usd`.
- **Each machine is a full factory**, not a thin worker: it needs the repo, an authenticated `claude` CLI, and the market data. Market data is already committed to the repo (the 15-name OHLCV fixtures), so a fresh clone is self-sufficient.
- **Dedup is eventual.** A machine dedups only against ideas it has pulled; two machines can independently produce similar ideas within a sync window. This is an accepted tradeoff of git-only coordination.

---

## 11. Testing

All tests are offline — no network, no real GitHub.

- **ID scheme** — `gen_<node>_<ts>` format; `pick_unused_strategy_id` `_2/_3` fallback still works.
- **Results shards** — `write_record` appends to the right shard; `read_records` unions multiple shard files; records sortable by `timestamp`.
- **Dedup shards** — `append_summary` writes timestamped lines; `read_tail` unions shards, returns the globally most-recent `n` oldest-first; legacy un-timestamped lines treated as oldest.
- **Registry auto-discovery** — only `gen_*.py` discovered; deterministic sorted order; `importlib.invalidate_caches()` invoked; a deliberately-broken `gen_*.py` fixture is skipped, not fatal; curated strategies still registered.
- **Sync** — `factory/sync.py` tested against a throwaway local git repo acting as the "remote": simulate two nodes each pushing their own shards, assert conflict-free rebase and that `sync_push` retry converges; assert a simulated push rejection is retried; assert a simulated unreachable remote logs and does not raise out of the loop.
- **Backwards compatibility** — with `[sync] enabled = false` and `node_id = "local"`, a full cycle behaves as the single-machine factory; `bootstrap()` correctly folds a legacy `results.json` + `dedup_log.txt`.

---

## 12. File layout

| File | Action |
|---|---|
| `factory/sync.py` | **create** — `bootstrap`, `sync_pull`, `sync_push` |
| `factory/settings_loader.py` | modify — `node_id`, `[sync]` section, `results_dir`/`dedup_dir` keys |
| `factory/config/settings.toml` | modify — new keys/section with safe defaults (`enabled=false`) |
| `factory/filesystem.py` | modify — `node_id` in ID scheme; remove `append_registry_entry` |
| `factory/cycle.py` | modify — `gen_<node>_<ts>` minting; drop the registry-append call |
| `factory/results.py` | modify — shard write, union read |
| `factory/dedup.py` | modify — timestamped shard write, union `read_tail`, legacy tolerance |
| `factory/loop.py` | modify — `sync_pull`/`sync_push` around each cycle |
| `backtester/strategies/registry.py` | modify — generated-strategy auto-discovery (curated registration unchanged) |
| `.gitignore` | modify — ensure `factory/data/_tmp/`, `factory/logs/`, `output/runs/` ignored |
| `factory/dashboard/server.py` | modify — one-line: pass `results_dir` to `read_records` (no logic change) |
| `factory/promote.py` | modify — one-line: pass `results_dir` to `read_records` (no logic change) |
| `factory/tests/test_sync.py` | **create** |
| `factory/tests/test_filesystem.py`, `test_results.py`, `test_dedup.py` | modify/extend — new ID scheme, shard write/union read |
| `backtester` registry test suite | modify/extend — auto-discovery coverage |

---

## 13. Out of scope

- **Live / real-time dedup** — would require a central service; explicitly rejected in favor of git-only.
- **A central coordinator, server, or database** — explicitly rejected.
- **Collaborators / untrusted contributors** — the design assumes all machines are the user's own and trusted; no auth model, no untrusted-code sandboxing beyond the factory's existing validation.
- **Resilience / failover guarantees** — machines are independent; one going down just stops its own contribution. No leader election, no heartbeats.
- **Automatic merge of `factory-pool` into `master`** — promoting pool strategies into `master` stays a deliberate human action.
- **Work partitioning** — machines are not assigned disjoint slices of the idea space; they draw slots independently and rely on eventual dedup.
- **Registry import performance at very large pool sizes** — importing thousands of generated modules will eventually be slow; that is a pre-existing characteristic (the explicit registry already imports them all) and is not addressed here.

---

## 14. Acceptance criteria

1. `node_id` is read from settings, validated against `^[a-z0-9][a-z0-9-]*$`, and defaults to `local`; a malformed value fails startup with a clear message.
2. Strategy IDs are `gen_<node_id>_<unix-second>`; derived strategy/config filenames are globally unique across machines.
3. `results.write_record` appends only to `factory/data/results/<node_id>.jsonl`; `read_records` returns the union of all shards.
4. `dedup.append_summary` writes timestamped lines to `factory/data/dedup/<node_id>.txt`; `read_tail(n)` returns the globally most-recent `n` summaries oldest-first, tolerating legacy un-timestamped lines.
5. `append_registry_entry` is gone; `backtester/strategies/registry.py` auto-discovers `gen_*.py` only, in sorted order, after `importlib.invalidate_caches()`; a module that fails to import is skipped with its filename and exception logged; curated strategies remain registered.
6. `factory/sync.py` provides `bootstrap`, `sync_pull`, `sync_push`: `bootstrap` creates `factory-pool` locally and pushes it to the remote on first run; `sync_pull` skips the cycle's sync on an unexpectedly dirty tree; `sync_push` is a logged no-op when nothing is staged and retries a non-fast-forward up to `push_retries` times until it converges; all sync failures are logged and never abort the loop.
7. With `[sync] enabled = false`, the factory behaves as the single-machine version; `bootstrap()` folds a legacy `results.json` + `dedup_log.txt` into this machine's shards idempotently.
8. The full factory test suite passes, including the new `test_sync.py` and the extended store/registry tests; sync tests use a local throwaway repo, not the network.
