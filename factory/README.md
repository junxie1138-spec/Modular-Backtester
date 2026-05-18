# Strategy Factory

An unattended loop that mass-produces trading-strategy ideas with `claude -p`, validates each one, runs it through the full backtest → optimize → WFO pipeline, and surfaces the survivors on a local dashboard and via Telegram. It wraps the Modular-Backtester and edits no backtester source file — generated strategies are picked up by registry auto-discovery.

This README covers running the factory on **one machine** and, in detail, running it as a **distributed multi-machine pool** coordinated through git.

---

## What one cycle does

```
pull pool  →  draw idea slots  →  claude -p generates a strategy
           →  static + functional validation
           →  write strategy .py + config .yaml
           →  backtest  →  optimize  →  (screen?)  →  WFO
           →  (held-out promotion on alternate tickers, if WFO clears the trigger)
           →  record result  →  Telegram alert if it clears threshold
           →  push pool
```

Every cycle writes one results record — `complete`, `complete (screened)`, or `failed` — to this machine's results shard. The `pull pool` / `push pool` steps are no-ops unless distributed mode is enabled.

**Critical framing:** at the factory's throughput, some strategies post OOS Sharpe > 1.0 on luck alone. WFO mitigates but does not eliminate multiple-comparisons risk. The dashboard "good" flag and the Telegram alert are **shortlist signals, not verdicts** — the held-out promotion gate (alternate tickers) is the first real filter, and a human review is still required before trusting any row.

---

## Install

From the backtester repo root (Python 3.11+):

```bash
pip install -e .[dev]     # backtester package + pytest      (quote as '.[dev]' on zsh/macOS)
pip install -e .[data]    # yfinance — needed by held-out promotion's default tickers
pip install Flask         # the dashboard's only extra runtime dependency
```

You also need the **`claude` CLI** authenticated on this machine and on `PATH`. The factory invokes it as a subprocess; `claude_cmd` defaults to `"claude"` (on Windows the `claude.CMD` npm shim is resolved automatically). Verify with `claude --version`.

---

## Single-machine quickstart

1. Open `factory/config/settings.toml` and set `[paths] backtester_root` to an absolute path if you will run the factory from a different working directory (`"."` is fine when you run from the repo root).
2. Put secrets in `factory/config/settings.local.toml` (gitignored — see [Configuration](#configuration)). For Telegram alerts, set `telegram_bot_token` and `telegram_chat_id` there.
3. Run the loop:
   ```bash
   python -m factory.loop
   ```
4. In a second terminal, run the dashboard:
   ```bash
   python -m factory.dashboard.server
   ```
5. Open <http://127.0.0.1:8787>.

The loop runs until `Ctrl-C` (graceful — it finishes the current cycle), or until a cycle limit is reached. `python -m factory.loop --seed 42` makes idea-slot draws reproducible; `--settings <path>` points at an alternate settings file.

### Choosing how many strategies to generate

Each cycle is one strategy attempt, so the cycle count *is* how many strategies a run generates. By default the loop runs forever (`[loop] max_cycles = 0` in `settings.toml`). To cap a run, pass `--max-cycles`:

```bash
python -m factory.loop --max-cycles 25   # run 25 cycles, then exit cleanly
```

- `--max-cycles N` runs **exactly N cycles** — N strategy attempts, whether each one ends `complete`, `complete (screened)`, or `failed` — and then the loop exits.
- `--max-cycles 0` means unlimited (the same as the default).
- The flag **overrides `[loop] max_cycles`** for that run only. Prefer it over editing `settings.toml` — `settings.toml` is tracked, and a modified tracked file makes distributed `sync_pull` skip every cycle (see [Troubleshooting](#troubleshooting)).
- A negative value is rejected before the loop starts.
- Omit the flag and the `[loop] max_cycles` setting applies instead.

`Ctrl-C` still stops a run early at any point — gracefully, after the current cycle finishes — regardless of `--max-cycles`.

With distributed mode off (the default), this is the whole factory — the per-machine ID scheme and sharded storage described below are still used, they are simply harmless on one machine.

---

## Configuration

Two files, both in `factory/config/`:

| File | Committed? | Holds |
|---|---|---|
| `settings.toml` | yes | All non-secret settings; safe defaults. |
| `settings.local.toml` | **no — gitignored** | Per-machine `node_id` and secrets (Telegram token, etc.). |

At load time `settings.local.toml` is **merged over** `settings.toml`, section by section. Anything you put in the local file wins. Never put a secret or a machine-specific value in `settings.toml`.

### `settings.toml` sections

| Section | Key settings |
|---|---|
| *(top level)* | `node_id` — this machine's identity (see distributed mode). Default `"local"`. |
| `[paths]` | `backtester_root` and the repo-relative dirs for strategies, configs, results shards, dedup shards, logs, tmp. |
| `[generation]` | `claude_cmd`, `claude_flags`, `generation_timeout_sec`. |
| `[stages]` | `stage_timeout_sec` — per-stage subprocess timeout. |
| `[alerts]` | `alert_threshold_metric` / `alert_threshold` (default `wfo.oos_sortino` > 1.0), Telegram credentials, `dashboard_base_url`. |
| `[loop]` | `inter_cycle_sleep_sec`, `max_cycles` (`0` = unlimited). |
| `[dashboard]` | `host`, `port`, `auto_refresh_sec`. |
| `[promotion]` | Held-out gate: `tickers`, `data_source`, `min_avg_sortino`, `trigger_metric`/`trigger_threshold`. |
| `[screening]` | Skip WFO when the best optimize score is below `min_optimize_score`. |
| `[sync]` | Distributed mode — see below. `enabled = false` by default. |

### Secrets and per-machine values — `settings.local.toml`

A ready-to-fill template is committed at `factory/config/settings.local.toml.example`. Copy it and edit:

```bash
cp factory/config/settings.local.toml.example factory/config/settings.local.toml
```

```toml
# factory/config/settings.local.toml  —  NEVER committed
node_id = "desk"

[alerts]
telegram_bot_token = "123456:AA..."
telegram_chat_id   = "-100..."

[sync]
enabled = true
```

`node_id` lives here precisely because it differs per machine and the file is already gitignored and per-machine. **`node_id` must be a top-level key, above every `[section]`** — in TOML a key written after a `[header]` belongs to that table, so a misplaced `node_id` is silently ignored. It must also match `^[a-z0-9][a-z0-9-]*$` (lowercase letters, digits, hyphens) or startup fails with a clear error.

---

## Distributed mode

Run the factory on several of your own machines at once, all contributing into **one shared strategy pool**, coordinated entirely through this git repository — no extra servers, services, or databases. N machines grow the pool roughly N× faster (and cost roughly N× the LLM spend).

### How it works

**The core rule: no two machines ever write the same file.** Hold that, and `git pull --rebase` is always conflict-free and `git push` always converges. Every machine-owned file is keyed by that machine's `node_id`:

| State | Single-machine | Distributed |
|---|---|---|
| Strategy IDs | `gen_<ts>` | `gen_<node_id>_<ts>` — e.g. `gen_desk_1778829071` |
| Strategy / config files | `strategies/gen_*.py`, `configs/wfo/gen_*.yaml` | same, but globally-unique names (node_id in the ID) |
| Results store | one `results/<node_id>.jsonl` shard | one shard **per machine**, union on read |
| Dedup log | one `dedup/<node_id>.txt` shard | one shard **per machine**, union on read |
| Registry | auto-discovery of `strategies/gen_*.py` | identical — no per-strategy file edit |

The pool is the set-union of every machine's files on a dedicated **`factory-pool` git branch**. "The pool" = all shards on that branch. Each machine appends only to its own shard, so shards never collide on push or pull.

### `[sync]` settings

```toml
[sync]
enabled      = true            # master switch — off by default
branch       = "factory-pool"  # shared pool branch
remote       = "origin"
push_retries = 5               # bounded retry on a non-fast-forward push
```

### One-time setup, on each machine

1. **Clone the repo and install** (see [Install](#install)). Each machine is a *full* factory — it needs the repo, an authenticated `claude` CLI, and market data. Market data ships committed (the OHLCV fixtures under `data/raw/`), so a fresh clone is self-sufficient.
2. **Give the machine a unique `node_id`.** Copy the committed template, then edit it:
   ```bash
   cp factory/config/settings.local.toml.example factory/config/settings.local.toml
   ```
   ```toml
   # in factory/config/settings.local.toml — node_id must be a TOP-LEVEL key,
   # above every [section], and match ^[a-z0-9][a-z0-9-]*$
   node_id = "desk"
   ```
   A malformed or duplicated `node_id` defeats the no-collision guarantee. Pick a distinct short name per machine (`desk`, `laptop`, `vps1`). The template is gitignored once copied, so editing it never dirties the working tree.
3. **Enable sync** — in that same `settings.local.toml`, set:
   ```toml
   [sync]
   enabled = true
   ```
   Set this in the gitignored `settings.local.toml`, **not** the tracked `settings.toml`. Editing the tracked file dirties the working tree, which makes `sync_pull` skip every cycle — and the factory then commits pool updates to the wrong branch.
4. **Ensure git can push without a prompt** — the loop runs unattended. Use an SSH key or a cached credential helper for `origin`.
5. **Run the preflight check** to confirm the machine is actually ready:
   ```bash
   python -m factory.scripts.preflight
   ```
   It verifies Python version, factory dependencies, settings + `node_id`, writable data dirs, the `[sync]` config, the `claude` CLI (resolvable *and* authenticated, via a trivial live call), git ≥ 2.28, and that the `[sync]` remote is reachable with non-interactive credentials. It exits `0` only when no check fails — review any `WARN`/`FAIL` lines before continuing. Use `--skip-claude-probe` to skip the live (token-spending) `claude` call, or `--skip-remote` when offline.
6. **Start the loop:** `python -m factory.loop`.

On the **first** machine to start, the loop's one-time `bootstrap()` step creates the `factory-pool` branch off `master` and **publishes it to the remote** (the one intentional remote-mutating bootstrap action — the pool cannot work until the branch is visible). Every machine after that simply tracks the existing remote branch. `bootstrap()` is idempotent and also folds any pre-existing single-file `results.json` / `dedup_log.txt` into this machine's shards.

### What each cycle does in distributed mode

- **`sync_pull` (before the cycle):** checks out `factory-pool`, then `git fetch` + `git pull --rebase`. This brings in every other machine's new strategies, configs, and shards. If the working tree has unexpected *tracked* changes, the pull is **skipped for that cycle** (a warning is logged) rather than risking a rejected rebase — generation still proceeds and the next clean cycle catches up.
- **The cycle runs**, writing only this machine's `node_id`-keyed files.
- **`sync_push` (after the cycle):** stages this machine's new strategy/config files and its two shards, commits as `factory(<node_id>): pool update`, and pushes. A non-fast-forward rejection (another machine pushed first) triggers `git pull --rebase` + retry, up to `push_retries` times. The rebase is always conflict-free, so retries converge immediately. If nothing new was produced, the push is a logged no-op.

**Sync failure never aborts generation.** Network down, auth expired, an unexpected conflict — every failure is logged and the loop continues. A machine that cannot reach GitHub keeps generating locally and catches up on its next successful sync. Dedup is therefore *eventual*: a machine dedups only against ideas it has already pulled, so two machines can briefly produce similar ideas within a sync window. That is the accepted tradeoff of git-only coordination.

### Watching the pool

Run the dashboard on any machine in the pool:

```bash
python -m factory.dashboard.server
```

Because the dashboard reads the **union of all shards** on the local checkout of `factory-pool`, after a `sync_pull` it shows the entire pool — every machine's strategies, the cumulative LLM **spend**, and the cumulative **token** consumption across all machines. No per-machine dashboard aggregation is needed; the git branch *is* the shared state.

### Promoting pool strategies into `master`

The `factory-pool` branch is a staging ground. Merging a vetted generated strategy back into `master` is a **deliberate human action** — there is no automatic merge. Review the candidate on the dashboard, confirm it cleared the held-out promotion gate, then cherry-pick or merge its files into `master` yourself.

### Mixed Windows + macOS pools

A pool can mix Windows and macOS machines. The repo ships a `.gitattributes` that pins the synced factory paths (generated strategies, configs, result/dedup shards) to LF line endings, so `sync_pull` never sees spurious line-ending churn and rebases stay conflict-free. No action needed — just give each machine a distinct `node_id`. (git 2.28+ is required, for `git init -b`; current macOS / Xcode git is new enough.)

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Startup fails: *invalid node_id* | `node_id` must match `^[a-z0-9][a-z0-9-]*$`. Set it in `settings.local.toml`. |
| `sync_pull: working tree has tracked changes; skipping` | You have uncommitted tracked edits on the checkout. Commit, stash, or revert them; the next cycle will sync. A common cause is editing the tracked `settings.toml` — move per-machine settings to `settings.local.toml`. |
| `sync_push: working tree is on '...', not the pool branch` | `sync_pull` skipped (dirty tree), so the loop never reached the pool branch. `sync_push` correctly refuses to commit to the wrong branch — resolve the tracked changes and sync resumes. |
| `sync_push: push still failing after N retries` | The remote is unreachable or auth expired. Generation continues; fix git auth and the next cycle catches up. Raise `push_retries` if your pool is very large and pushes collide often. |
| Two machines produced near-identical ideas | Expected within a sync window — dedup is eventual. It self-corrects as shards propagate. |
| A generated strategy is missing from a run | A `gen_*.py` that fails to import is skipped (auto-discovery logs its filename and the exception). Check the factory log. |

---

## Operational scripts

```bash
python -m factory.scripts.preflight                      # distributed-readiness check (run before joining a pool)
python -m factory.scripts.endurance_check --cycles 100   # validate N unattended cycles
python -m factory.scripts.telegram_smoke                 # verify Telegram credentials send
```

`preflight` is the readiness gate for the distributed factory — run it on each machine before enabling `[sync]`. `endurance_check` runs real backtest/optimize/WFO cycles with a stubbed `claude -p`, exercising the heavy local pipeline. Neither touches the network destructively: `preflight`'s remote check is a read-only `git ls-remote`, and all sync *tests* use a throwaway local repo.

Logs rotate under `factory/logs/factory.log` (10 MB × 5 backups) and also mirror to stderr for interactive runs.

---

## Upgrading from an earlier version

Ran the factory before these changes? Here is how to update and what carries over.

### How to update

1. **Pull the new code.** Single-machine: `git pull` on the branch you run from. Distributed: the loop runs `factory-pool`'s code, so update *that* branch — on one machine `git checkout factory-pool && git merge master && git push origin factory-pool`; every other machine picks the update up on its next `sync_pull`.
2. **Refresh dependencies** — re-run the [Install](#install) steps. `Flask` and the `[data]` extra (yfinance, used by held-out promotion) may be newer requirements than when you first set up.
3. **Run the preflight check** — `python -m factory.scripts.preflight` — to confirm the machine is still good.

### Is my progress saved? — yes

An upgrade discards no generated work:

- **Strategy and config files** (`strategies/gen_*.py`, `configs/wfo/gen_*.yaml`) are plain files on disk — untouched. The registry auto-discovers them, including older `gen_<timestamp>.py` ids minted before the `gen_<node_id>_<timestamp>` scheme.
- **Results records are forward-compatible.** Token tracking adds a `generation_tokens` field to *new* records only; older records have no such field and are read as "no token data" — the detail view shows `n/a` for them and they contribute `0` to the cumulative-tokens total. No migration, no data loss.

### If you ran a much older (single-file) version

A pre-distributed factory kept one `factory/data/results.json` and one `factory/data/dedup_log.txt`. The current factory uses per-machine shard directories — `factory/data/results/<node_id>.jsonl` and `factory/data/dedup/<node_id>.txt`.

- **Enabling distributed mode:** the first `bootstrap()` (run automatically at loop start) folds those legacy files into this machine's shards for you — a one-time verbatim copy, skipped if the shard already exists.
- **Staying single-machine** (`[sync] enabled = false`): `bootstrap()` is a no-op, so do the one-time copy yourself (create the `results/` and `dedup/` directories first if they do not exist):
  ```bash
  cp factory/data/results.json   factory/data/results/local.jsonl    # "local" = default node_id
  cp factory/data/dedup_log.txt  factory/data/dedup/local.txt
  ```
  The formats are compatible — results is JSONL either way, and old un-timestamped dedup lines are tolerated (treated as oldest). Skip the copy and only the dashboard history of those old runs is lost; the strategy files themselves are unaffected.

### Moving to distributed mode

Put `node_id` and `[sync] enabled` in the gitignored `settings.local.toml` (copy `settings.local.toml.example`) — **not** in the tracked `settings.toml`. A tracked-but-modified `settings.toml` permanently dirties the working tree, which makes `sync_pull` skip every cycle and the factory commit pool updates to the wrong branch. See [Configuration](#configuration).

---

## Tests

```bash
python -m pytest factory/tests -q
```

The factory has a fast unit suite plus slower integration tests. Slow tests (Tier 2 functional validation, integration smoke) are marked `@pytest.mark.slow` — add `-m "not slow"` to skip them or `-m slow` to run only those. All sync tests use a throwaway local git repo as the "remote"; nothing in the suite touches the network.

---

## Design references

- Distributed factory — [`docs/superpowers/specs/2026-05-16-distributed-factory-design.md`](../docs/superpowers/specs/2026-05-16-distributed-factory-design.md)
- Token tracking — [`docs/superpowers/specs/2026-05-16-factory-token-tracking-design.md`](../docs/superpowers/specs/2026-05-16-factory-token-tracking-design.md)
- Original factory build — [`docs/superpowers/plans/2026-05-15-strategy-factory-v020.md`](../docs/superpowers/plans/2026-05-15-strategy-factory-v020.md)
