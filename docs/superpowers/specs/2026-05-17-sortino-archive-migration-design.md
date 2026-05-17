# Sortino archive migration — design

**Date:** 2026-05-17
**Status:** Approved (brainstorm complete, ready for implementation planning)

## Problem

The strategy factory's alert and promotion metrics were re-pointed from
Sharpe to **OOS Sortino** (commits `e180469`, `9100d31`, `72aad56`, …), and
generated WFO configs now optimise in-sample on sortino. But strategy records
produced *before* that change are stranded: their `wfo` block carries
`oos_sharpe` and **no `oos_sortino`**, and their `optimize.objective` is
`"sharpe"`. The dashboard, the alert path, and promotion all now key on
`oos_sortino` — so archived strategies are judged by a field they don't have.

Two consequences:

1. **Inconsistent evaluation.** An archived strategy and a new one are not
   comparable — one has `oos_sortino`, the other doesn't.
2. **No retroactive promotion.** An archived strategy that *would* clear the
   sortino promotion threshold was never given the held-out promotion stage,
   because at generation time it was judged on Sharpe.

This must also work across the distributed factory pool: every machine's
archive should converge to the new format with no manual operator step.

## Goal

Every archived strategy record carries an `oos_sortino` and is judged by the
same OOS-Sortino system as new strategies; archived strategies that clear the
promotion threshold are retroactively promoted; this happens automatically on
every machine in the pool, with no operator coordination.

## Approach

**Re-score, not re-run.** The on-disk WFO bundle for every past run already
contains `oos_summary.sortino` in its `summary.json` — the factory simply
never extracted it into the record. So `oos_sortino` is recovered by a pure
**read** of the existing bundle; no optimise/WFO recompute. The strategy keeps
its original Sharpe-optimised parameters. Records where re-optimising on
sortino *might* change the verdict are **flagged** (`needs_rerun`) but never
re-run — a true re-optimisation is explicitly out of scope.

**Per-shard, distributed-safe.** Strategy records live in per-node shards,
`results/<node_id>.jsonl`; each machine is the sole writer of its own shard,
and git sync is conflict-free precisely because no machine ever touches
another's shard. The migration honours that invariant exactly: it only ever
rewrites **this machine's own shard**. Migrated shards propagate on the normal
git push/pull. "Other machines run it too" therefore needs no new sync logic —
each machine migrates its own shard the next time it starts the factory.

**Auto at startup, retro-promotion queued.** The factory loop runs one
idempotent migration pass at startup. Retroactive promotion is real compute
(~3 held-out WFO subprocess runs per qualifying strategy), so it is *queued*:
the migration marks qualifying records, and the loop drains one queued
retro-promotion per cycle — startup stays fast and the compute spreads
alongside normal generation.

## Architecture & components

### New module: `factory/sortino_migration.py`

One responsibility — migrate this machine's own shard. Network-free except the
retro-promotion drain (which shells out to the existing WFO stage via
`promote_strategy`). Two public entry points:

- `migrate_shard(settings)` — one idempotent pass over `results/<node_id>.jsonl`.
- `drain_one_retro_promotion(settings)` — run at most one queued
  retro-promotion, rewriting the affected record.

Both read and rewrite only `results/<node_id>.jsonl`. A JSONL shard cannot be
edited line-in-place, so each call rewrites the whole shard file — still
sole-writer, still conflict-free.

### Record additions

A migrated `status: complete` record gains, inside its existing `wfo` block:

- `wfo.oos_sortino` — `float`, read from the WFO bundle's
  `summary.json` → `oos_summary.sortino`.

and a new top-level block:

```json
"sortino_migration": {
  "migrated_at": "<iso8601 UTC>",
  "needs_rerun": <bool>,
  "state": "pending" | "done" | "n/a"
}
```

`oos_sortino` sits alongside the `oos_sharpe` the record already has — the
same shape `parse_wfo_summary` produces for new records, so a migrated record
is indistinguishable from a natively-sortino one to every downstream reader.

## `migrate_shard` behaviour

Read every record in the local shard. For each:

1. **Skip** records that are not `status == "complete"`, or have no `wfo`
   block — nothing to backfill, no marker added.
2. **Skip** (idempotency) records whose `wfo` block already contains
   `oos_sortino`. This covers both natively-sortino records and
   already-migrated ones, so a second `migrate_shard` run is a fast no-op.
3. **Backfill.** Read `wfo.run_bundle_path` → `summary.json` →
   `oos_summary.sortino`; write it as `wfo.oos_sortino`. If the bundle
   directory or its `summary.json` is missing, or `summary.json` carries no
   `oos_summary.sortino` value (an operator cleaned `output/runs/`, which is
   gitignored and local-only, or the bundle predates sortino in the WFO
   summary), leave the record **untouched**, log a warning, and continue — no
   recompute. Such a record is simply re-examined on the next startup.
4. **Add the `sortino_migration` block:**
   - `migrated_at` — current UTC timestamp, ISO 8601.
   - `needs_rerun` — `true` iff `oos_sharpe` and `oos_sortino` fall on
     **opposite sides of `promotion.trigger_threshold`** (i.e. the
     Sharpe→Sortino swap flips this strategy's promote/no-promote standing).
     Computed unconditionally, even when promotion is disabled.
   - `state` — see the state machine below.

If any record changed, rewrite the shard file (one JSON object per line,
preserving record order).

### The `state` machine

| `state`   | Meaning |
|-----------|---------|
| `pending` | Eligible and awaiting retro-promotion. |
| `done`    | Retro-promotion is present or has been performed. |
| `n/a`     | Not eligible, promotion disabled, or impossible to process. |

`migrate_shard` assigns the initial `state` per record:

- **`done`** — the record already has a `promotion` block. It is already past
  the retro-promotion stage; nothing to queue. (`needs_rerun` is still
  computed and stored.)
- **`pending`** — no `promotion` block, promotion is enabled in settings, and
  `oos_sortino >= promotion.trigger_threshold`.
- **`n/a`** — otherwise: promotion disabled, or `oos_sortino` below the
  trigger threshold.

**Invariant — `state` is assigned exactly once.** `migrate_shard` is
idempotent on `wfo.oos_sortino`: a record that already carries it is skipped
entirely on every later run, so its `state` is set once, at first migration,
and never revisited by `migrate_shard`. In particular, **a record migrated
while promotion was disabled receives `state: "n/a"` and does not transition
`n/a → pending` if promotion is later enabled** — `migrate_shard` does not
re-scan already-migrated records when promotion settings change. This is a
deliberate v1 limitation, not an omission; an operator who enables promotion
after a migration and wants the backlog reconsidered must trigger that
explicitly (no such trigger exists in v1). The only `state` transitions after
first migration are the `pending → done` / `pending → n/a` moves made by
`drain_one_retro_promotion`.

`drain_one_retro_promotion` advances `state`:

- `pending` → `done` once promotion has run.
- `pending` → `n/a` if the strategy cannot be processed (canonical config
  missing — see below).

A `pending` record therefore always resolves to a terminal `done` or `n/a`;
it is never stuck.

## Retro-promotion queue (`drain_one_retro_promotion`)

Once per cycle:

1. Read the local shard; find the **first** record with
   `sortino_migration.state == "pending"`. None → no-op return.
2. Locate the strategy's canonical WFO config at
   `configs_dir/<strategy_id>.yaml` (configs are committed and synced
   pool-wide, so this is available even for a strategy generated on another
   machine — but the record being migrated is always in *this* machine's
   shard). If the config is missing, set `state: "n/a"`, log, rewrite the
   shard, return.
3. Call the existing `promote_strategy(...)` from `factory/promote.py` — the
   held-out promotion stage — with `optimized_params` taken from the record's
   `optimize.best_params` and the settings-derived arguments (`promotion_cfg`,
   `tmp_dir`, `output_runs_dir`, `stage_timeout_sec`, `backtester_root`,
   `build_report_path`), exactly as `factory/cycle.py` invokes it.
4. Write the returned `PromotionResult` (as a dict) into the record's
   `promotion` field, set `sortino_migration.state: "done"`, and rewrite the
   shard.

At most one retro-promotion runs per cycle, so the held-out compute is spread
across cycles rather than blocking startup. The queue is drained purely by
scanning the shard for `pending` — there is no separate queue file.

## Loop integration (`factory/loop.py`)

`run_loop` already calls `bootstrap(settings)` once at startup, then loops
`sync_pull` → `run_cycle` → `sync_push`.

- **Startup:** immediately after the `bootstrap` call, invoke
  `migrate_shard(settings)`, wrapped in `try/except` and logging on failure —
  consistent with how `bootstrap`/`sync_pull`/`sync_push` are wrapped so that a
  failure logs and the factory continues. The migration runs regardless of
  `sync.enabled` (a standalone machine still has a shard and old records).
- **Per cycle:** invoke `drain_one_retro_promotion(settings)` between
  `run_cycle` and `sync_push`, similarly wrapped, so the rewritten shard rides
  out on the same cycle's push.

## Distributed behaviour

Every write the migration makes — the `oos_sortino` backfill and the
retro-promotion `promotion` block — lands in `results/<node_id>.jsonl`, the
file this machine already solely owns. Git sync is unchanged: `sync_push`
already commits `results_dir`, so the migrated shard propagates with the
normal pool update; `sync_pull` rebases conflict-free because no other machine
writes this shard. No new sync code.

**One benign startup artifact.** The startup `migrate_shard` dirties the shard
(a tracked file). `sync_pull`'s existing dirty-tree guard then skips the pull
for the *first* cycle; that cycle's `sync_push` commits the migration, and
sync resumes normally from cycle 2. This is self-healing and requires no
special handling — it is noted here only so the one-cycle "sync_pull skipped"
log line is expected, not alarming.

## Scope boundaries

**In scope:** `oos_sortino` backfill from existing bundles; the
`sortino_migration` block; the `needs_rerun` flag; queued retroactive
promotion; auto-trigger at factory startup; per-shard distributed safety.

**Out of scope:**

- **Re-running optimise/WFO under the sortino objective** — the "deep"
  migration. Records where it might matter are *flagged* via `needs_rerun`;
  the re-optimisation itself is a separate, future, manual decision.
- **Dashboard UI** for the `needs_rerun` flag or the migration state.
- **Migrating the legacy single-file `factory/data/results.json`** into the
  shard store — that remains `sync.py`'s `_fold_legacy_stores` responsibility.
  The migration operates on the shard store (`results/<node_id>.jsonl`), which
  is the live record store every code path already reads and writes.
- **Re-sending alerts** for strategies that now clear the *alert* threshold
  under sortino. Only retroactive *promotion* was requested; replaying old
  Telegram alerts would be noise.

## Testing

- **Unit — backfill:** a synthetic record + a fake bundle directory whose
  `summary.json` carries `oos_summary.sortino` → `migrate_shard` writes
  `wfo.oos_sortino` and a `sortino_migration` block.
- **Unit — missing bundle:** a record whose `run_bundle_path` does not exist →
  record left untouched, no `oos_sortino`, no marker; warning logged.
- **Unit — `needs_rerun`:** Sharpe and Sortino on the same side of the
  threshold → `false`; on opposite sides → `true`.
- **Unit — state assignment:** below-threshold → `n/a`; above-threshold +
  promotion enabled + no `promotion` block → `pending`; record with an
  existing `promotion` block → `done`; promotion disabled → `n/a`.
- **Unit — idempotency:** run `migrate_shard` twice; the second run rewrites
  nothing (every record already has `wfo.oos_sortino`).
- **Unit — skip rules:** `status: failed` and no-`wfo` records are left
  untouched and unmarked.
- **Unit — `drain_one_retro_promotion`:** with `promote_strategy` mocked, a
  `pending` record gets a `promotion` block and `state: "done"`; a record
  whose canonical config is absent gets `state: "n/a"`; with no `pending`
  record the call is a no-op.
- **Integration:** `loop.py` calls `migrate_shard` once at startup and
  `drain_one_retro_promotion` once per cycle.
- **Regression:** the existing factory test suite stays green.

## Risks & open items

- **Bundle availability.** Backfill depends on the WFO bundle still being on
  disk. A machine has bundles for the strategies *it* generated (its own
  shard), so per-shard migration is self-consistent — but an operator who
  pruned `output/runs/` loses the ability to backfill those records. They are
  skipped and logged, not recomputed; this is an accepted limitation of the
  re-score approach.
- **Promotion toggled on after migration.** Covered as an explicit invariant
  in the state-machine section above — a record migrated under disabled
  promotion stays `n/a` and is not reconsidered if promotion is later enabled.
  Surfaced there rather than here so it is read as designed behaviour, not an
  accidental gap.
- **`needs_rerun` is advisory only.** It marks records for a possible future
  re-optimisation; nothing acts on it automatically. That is deliberate.
