# Screen Out Hopeless Strategies Before WFO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the optimize stage succeeds, skip the expensive WFO stage when the best in-sample score is below a configurable floor, because such a strategy cannot clear the OOS shortlist gate anyway.

**Architecture:** A new optional `[screening]` settings section gates a check inserted at the top of the `wfo` iteration of the existing stage loop in `factory/cycle.py`. A screened cycle still records `status=complete` (intentional skip, not failure) but with `wfo=null`, `promotion=null`, `screened_out=true`, and a human-readable `screen_reason`. The results-record schema gains two always-present keys so the dashboard stays uniform across all record types.

**Tech Stack:** Python 3.11+ (`tomllib`, frozen `dataclass`), pytest, Flask + Jinja2 templates, vanilla JS.

**Constraints:**
- Working directory: `C:/Users/aiden/Documents/VScode_Work/Backtester`. Branch: `v0.4.0-mean-reversion-atr`.
- Do **NOT** modify any file under `backtester/`. Do **NOT** push.
- One atomic commit at the very end (Task 6) — no per-task commits.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `factory/config/settings.toml` | Real settings | Add `[screening]` section |
| `factory/settings_loader.py` | Typed settings model + loader | Add `ScreeningCfg`, wire into `Settings` + `load_settings` |
| `factory/results.py` | Results-record builders | Add `screened_out` / `screen_reason` keys to both builders |
| `factory/cycle.py` | One full cycle orchestration | Insert the screening gate; guard WFO/promotion dereferences |
| `factory/dashboard/server.py` | Dashboard aggregation | Add `screened` counter |
| `factory/dashboard/templates/overview.html` | Overview table + counters | Show `Screened` counter + `complete (screened)` status |
| `factory/dashboard/static/overview.js` | Overview live refresh | Render `complete (screened)`; update `#c-screened` |
| `factory/dashboard/templates/detail.html` | Per-strategy detail page | Explain skipped WFO section |
| `factory/tests/conftest.py` | Test settings fixture | Add `[screening]` to `tmp_settings_file` TOML |
| `factory/tests/test_cycle.py` | Cycle integration tests | Add `test_screened_out_skips_wfo_and_promotion` |

The integration test in `test_cycle.py` is the driving failing test: it exercises settings → results → cycle end-to-end, so it cannot pass until Tasks 2–4 are done. Dashboard changes (Task 5) have no automated test and are verified by inspection.

---

### Task 1: Write the driving failing test

**Files:**
- Modify: `factory/tests/conftest.py` (append a `[screening]` section to the `tmp_settings_file` TOML, after the `[dashboard]` block)
- Test: `factory/tests/test_cycle.py` (append a new test function at end of file)

**Why this test fails first:** `run_cycle` will reference `s.screening.enabled`, but `Settings` has no `screening` field yet — the test raises `AttributeError` until Task 2 lands, then exercises the gate added in Task 4.

- [ ] **Step 1: Add `[screening]` to the test settings fixture**

In `factory/tests/conftest.py`, the `tmp_settings_file` TOML string currently ends with the `[dashboard]` block (the last three lines before the closing `"""`):

```
        [dashboard]
        host             = "127.0.0.1"
        port             = 8787
        auto_refresh_sec = 10
        """
```

Replace that with (adds `[screening]` before the closing `"""`):

```
        [dashboard]
        host             = "127.0.0.1"
        port             = 8787
        auto_refresh_sec = 10

        [screening]
        enabled            = true
        min_optimize_score = 1.3
        """
```

Note: the existing `test_complete_cycle_writes_files_registry_record` stubs optimize with `best_score=1.3`. The gate fires only when `best_score < min_optimize_score`; `1.3 < 1.3` is `False`, so the screen does **not** fire there and that test still runs WFO unchanged.

- [ ] **Step 2: Add the new test to `test_cycle.py`**

Append this function to the end of `factory/tests/test_cycle.py`:

```python
def test_screened_out_skips_wfo_and_promotion(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "screened test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    from factory.stages import StageResult
    bt = StageResult(stage="backtest",
                     parsed={"sharpe": 0.4, "total_return": 0.05, "max_drawdown": -0.05,
                             "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p1"},
                     bundle_path=Path("p1"), raw_summary={})
    # best_score 0.5 is below the 1.3 floor -> screen fires.
    opt = StageResult(stage="optimize",
                      parsed={"best_params": {"size": 1.0}, "objective": "sharpe",
                              "best_score": 0.5, "run_bundle_path": "p2"},
                      bundle_path=Path("p2"), raw_summary={})

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage", return_value=bt), \
         mock.patch("factory.cycle.run_optimize_stage", return_value=opt), \
         mock.patch("factory.cycle.run_wfo_stage") as run_wfo_stage:
        outcome = run_cycle(s, rng=random.Random(0))

    # WFO stage was never invoked.
    run_wfo_stage.assert_not_called()

    assert outcome.status == "complete"
    assert outcome.failed_stage is None
    assert outcome.record["screened_out"] is True
    assert outcome.record["wfo"] is None
    assert outcome.record["promotion"] is None
    assert "0.500" in outcome.record["screen_reason"]
```

- [ ] **Step 3: Run the new test — verify it FAILS**

Run: `python -m pytest factory/tests/test_cycle.py::test_screened_out_skips_wfo_and_promotion -q`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'screening'` (raised inside `run_cycle`).

---

### Task 2: Settings — `ScreeningCfg`

**Files:**
- Modify: `factory/config/settings.toml`
- Modify: `factory/settings_loader.py`

- [ ] **Step 1: Add the `[screening]` section to `settings.toml`**

In `factory/config/settings.toml`, append after the `[promotion]` block (currently the last block, ending with `trigger_threshold = 1.0`):

```toml

[screening]
enabled            = true
min_optimize_score = 1.3
```

- [ ] **Step 2: Add the `ScreeningCfg` dataclass**

In `factory/settings_loader.py`, insert this dataclass immediately after the `PromotionCfg` class (after its closing field `trigger_threshold: float` and the blank line), before the `Settings` class:

```python
@dataclass(slots=True, frozen=True)
class ScreeningCfg:
    enabled: bool
    min_optimize_score: float
```

- [ ] **Step 3: Add `screening` to the `Settings` dataclass**

In `factory/settings_loader.py`, the `Settings` dataclass currently ends with `promotion: PromotionCfg`. Add a `screening` field after it:

```python
@dataclass(slots=True, frozen=True)
class Settings:
    paths: Paths
    generation: GenerationCfg
    stages: StagesCfg
    alerts: AlertsCfg
    loop: LoopCfg
    dashboard: DashboardCfg
    promotion: PromotionCfg
    screening: ScreeningCfg
```

- [ ] **Step 4: Parse `[screening]` in `load_settings`**

In `factory/settings_loader.py`, `load_settings` currently sets `pr = raw.get("promotion", {}) or {}` just before the `return Settings(...)`. Add a sibling line right after it:

```python
    pr = raw.get("promotion", {}) or {}
    sc = raw.get("screening", {}) or {}
```

Then, in the `return Settings(...)` call, add the `screening=` argument as the last argument, after the `promotion=PromotionCfg(...)` block's closing `),`:

```python
        promotion=PromotionCfg(
            enabled=bool(pr.get("enabled", False)),
            tickers=tuple(pr.get("tickers", ())),
            data_source=str(pr.get("data_source", "yfinance")),
            min_avg_sharpe=float(pr.get("min_avg_sharpe", 0.7)),
            trigger_metric=str(pr.get("trigger_metric", "wfo.oos_sharpe")),
            trigger_threshold=float(pr.get("trigger_threshold", 1.0)),
        ),
        screening=ScreeningCfg(
            enabled=bool(sc.get("enabled", False)),
            min_optimize_score=float(sc.get("min_optimize_score", 1.3)),
        ),
    )
```

`enabled` defaults to `False` when the `[screening]` section is absent, so any `settings.toml` without it is unaffected.

- [ ] **Step 5: Verify settings load cleanly**

Run: `python -c "from factory.settings_loader import load_settings; from pathlib import Path; s = load_settings(Path('factory/config/settings.toml')); print(s.screening)"`
Expected: `ScreeningCfg(enabled=True, min_optimize_score=1.3)`

---

### Task 3: Results record schema

**Files:**
- Modify: `factory/results.py`

Both `build_record` and `build_failed_record` must emit `screened_out` and `screen_reason` keys so every record (complete, failed, screened) carries the same shape.

- [ ] **Step 1: Add params + keys to `build_record`**

In `factory/results.py`, `build_record`'s signature currently ends with `promotion: Optional[Mapping[str, Any]] = None,`. Add two keyword params after it:

```python
def build_record(
    *,
    strategy_id: str,
    timestamp: str,
    slots: Mapping[str, str],
    idea: Mapping[str, Any],
    generation_cost_usd: float,
    backtest: Optional[Mapping[str, Any]],
    optimize: Optional[Mapping[str, Any]],
    wfo: Optional[Mapping[str, Any]],
    alerted: bool,
    promotion: Optional[Mapping[str, Any]] = None,
    screened_out: bool = False,
    screen_reason: Optional[str] = None,
) -> Record:
```

In its returned dict, add the two keys after `"promotion": ...,` and before `"alerted": ...,`:

```python
        "wfo": dict(wfo) if wfo is not None else None,
        "promotion": dict(promotion) if promotion is not None else None,
        "screened_out": bool(screened_out),
        "screen_reason": screen_reason,
        "alerted": bool(alerted),
    }
```

- [ ] **Step 2: Add constant keys to `build_failed_record`**

`build_failed_record` gains **no new params** — only two constant keys in its returned dict. In `factory/results.py`, in the `build_failed_record` returned dict, add after `"promotion": ...,` and before `"alerted": False,`:

```python
        "wfo": dict(wfo) if wfo is not None else None,
        "promotion": dict(promotion) if promotion is not None else None,
        "screened_out": False,
        "screen_reason": None,
        "alerted": False,
    }
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `python -c "import factory.results"`
Expected: no output, exit 0.

---

### Task 4: Cycle screening gate

**Files:**
- Modify: `factory/cycle.py`

This is the delicate task. The gate sits at the top of the `wfo` iteration of the existing stage loop; the post-loop code is adjusted so `wfo` may legitimately be `None`.

- [ ] **Step 1: Add screening state before the stage loop**

In `factory/cycle.py`, the stage loop section currently begins:

```python
    # Step 11-13: run the three stages sequentially.
    bt: Optional[StageResult] = None
    opt: Optional[StageResult] = None
    wfo: Optional[StageResult] = None
    for stage_name, runner in (
```

Insert two state variables after the `wfo` declaration:

```python
    # Step 11-13: run the three stages sequentially.
    bt: Optional[StageResult] = None
    opt: Optional[StageResult] = None
    wfo: Optional[StageResult] = None
    screened_out = False
    screen_reason: Optional[str] = None
    for stage_name, runner in (
```

- [ ] **Step 2: Insert the screening gate at the top of the loop body**

In `factory/cycle.py`, the loop body currently starts directly with `try:`:

```python
    for stage_name, runner in (
        ("backtest", run_backtest_stage),
        ("optimize", run_optimize_stage),
        ("wfo", run_wfo_stage),
    ):
        try:
            result = runner(
```

Insert the gate between the `):` and `try:`:

```python
    for stage_name, runner in (
        ("backtest", run_backtest_stage),
        ("optimize", run_optimize_stage),
        ("wfo", run_wfo_stage),
    ):
        if stage_name == "wfo" and s.screening.enabled:
            assert opt is not None
            best_score = opt.parsed.get("best_score")
            if best_score is not None and best_score < s.screening.min_optimize_score:
                screened_out = True
                screen_reason = (
                    f"optimize best_score {best_score:.3f} < floor "
                    f"{s.screening.min_optimize_score:.3f}"
                )
                log.info("cycle id=%s screened out before WFO: %s",
                         strategy_id, screen_reason)
                break
        try:
            result = runner(
```

The existing `try/except StageError` and the `if stage_name == "backtest": ... elif ... else:` result assignment stay **unchanged**.

- [ ] **Step 3: Relax the post-loop assertion**

In `factory/cycle.py`, the assertion after the loop is currently:

```python
    assert bt is not None and opt is not None and wfo is not None
```

Change it to (drop the `wfo` clause — `wfo` is legitimately `None` when screened):

```python
    assert bt is not None and opt is not None
```

- [ ] **Step 4: Guard the promotion block against `wfo is None`**

In `factory/cycle.py`, the promotion block currently dereferences `wfo.parsed` unconditionally. Its guard is:

```python
    promotion_dict: Optional[dict[str, Any]] = None
    if s.promotion.enabled:
        provisional = {
            "backtest": bt.parsed, "optimize": opt.parsed, "wfo": wfo.parsed,
        }
```

Change the `if` to also require `wfo is not None`:

```python
    promotion_dict: Optional[dict[str, Any]] = None
    if s.promotion.enabled and wfo is not None:
        provisional = {
            "backtest": bt.parsed, "optimize": opt.parsed, "wfo": wfo.parsed,
        }
```

- [ ] **Step 5: Make the `build_record` call screening-aware**

In `factory/cycle.py`, the `build_record(...)` call currently is:

```python
    # Step 14-15: build complete record.
    rec = build_record(
        strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
        generation_cost_usd=cost,
        backtest=bt.parsed, optimize=opt.parsed, wfo=wfo.parsed,
        promotion=promotion_dict,
        alerted=False,  # patched below after maybe_send_alert
    )
```

Change `wfo=wfo.parsed` to be conditional, and pass the screening fields:

```python
    # Step 14-15: build complete record.
    rec = build_record(
        strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
        generation_cost_usd=cost,
        backtest=bt.parsed, optimize=opt.parsed,
        wfo=wfo.parsed if wfo is not None else None,
        promotion=promotion_dict,
        screened_out=screened_out,
        screen_reason=screen_reason,
        alerted=False,  # patched below after maybe_send_alert
    )
```

- [ ] **Step 6: Guard the final log line**

In `factory/cycle.py`, the final log line before the `return` is currently:

```python
    write_record(paths.results_store, rec)
    log.info("cycle id=%s complete oos_sharpe=%s alerted=%s",
             strategy_id, wfo.parsed.get("oos_sharpe"), rec["alerted"])
```

Change it to guard the `wfo` dereference and report the screening state:

```python
    write_record(paths.results_store, rec)
    log.info("cycle id=%s complete oos_sharpe=%s screened=%s alerted=%s",
             strategy_id,
             wfo.parsed.get("oos_sharpe") if wfo is not None else "n/a",
             screened_out, rec["alerted"])
```

The cycle status stays `"complete"` for a screened cycle — screening is an intentional skip, not a failure. The `return CycleOutcome(status="complete", ...)` line is unchanged.

- [ ] **Step 7: Run the driving test — verify it PASSES**

Run: `python -m pytest factory/tests/test_cycle.py::test_screened_out_skips_wfo_and_promotion -q`
Expected: PASS.

- [ ] **Step 8: Run the full non-slow suite — verify all green**

Run: `python -m pytest factory/tests -q -m "not slow"`
Expected: all tests pass, including the pre-existing `test_complete_cycle_writes_files_registry_record` (its `best_score=1.3` does not trip the `< 1.3` floor, so WFO still runs there).

---

### Task 5: Dashboard

**Files:**
- Modify: `factory/dashboard/server.py`
- Modify: `factory/dashboard/templates/overview.html`
- Modify: `factory/dashboard/static/overview.js`
- Modify: `factory/dashboard/templates/detail.html`

No automated tests here — verified by inspection and (optionally) a manual dashboard run.

- [ ] **Step 1: Add the `screened` counter to `_aggregate`**

In `factory/dashboard/server.py`, `_aggregate` currently initializes counters and loops over records. Add a `screened` counter. Initialize it next to `promoted`:

```python
    above_threshold = 0
    promoted = 0
    promotion_attempted = 0
    screened = 0
    cumulative_spend = 0.0
```

In the `elif r.get("status") == "complete":` branch, count screened records:

```python
        elif r.get("status") == "complete":
            if r.get("screened_out"):
                screened += 1
            val = extract_metric(r, threshold_metric)
            if val is not None and val > threshold:
                above_threshold += 1
```

Add `"screened": screened,` to the returned dict, next to `"promoted"`:

```python
        "above_threshold": above_threshold,
        "promotion_attempted": promotion_attempted,
        "promoted": promoted,
        "screened": screened,
        "cumulative_spend_usd": cumulative_spend,
```

- [ ] **Step 2: Add the `Screened` counter + status text to `overview.html`**

In `factory/dashboard/templates/overview.html`, the counters block currently has a `Promoted` span. Add a `Screened` span right after it:

```html
    <span>Promoted: <strong id="c-promoted">{{ summary.promoted }}</strong> / {{ summary.promotion_attempted }} attempted</span>
    <span>Screened: <strong id="c-screened">{{ summary.screened }}</strong></span>
    <span>Cumulative spend: <strong id="c-spend">${{ "%.2f"|format(summary.cumulative_spend_usd) }}</strong></span>
```

In the status `<td>`, the cell is currently:

```html
        <td>{% if r.status == 'failed' %}<span class="failed-stage">failed: {{ r.failed_stage }}</span>{% else %}{{ r.status }}{% endif %}</td>
```

Change the `{% else %}` arm to an `{% elif %}`:

```html
        <td>{% if r.status == 'failed' %}<span class="failed-stage">failed: {{ r.failed_stage }}</span>{% elif r.screened_out %}complete (screened){% else %}{{ r.status }}{% endif %}</td>
```

- [ ] **Step 3: Render `complete (screened)` and update `#c-screened` in `overview.js`**

In `factory/dashboard/static/overview.js`, the `cells` array in `rowFor` currently has the status cell as:

```javascript
      rec.status === "failed"
        ? `<span class="failed-stage">failed: ${rec.failed_stage}</span>`
        : rec.status,
```

Change the non-failed arm to handle screened rows:

```javascript
      rec.status === "failed"
        ? `<span class="failed-stage">failed: ${rec.failed_stage}</span>`
        : (rec.screened_out ? "complete (screened)" : rec.status),
```

In `refresh()`, the counter-update block currently ends with the `c-spend` line. Add a `c-screened` update before it:

```javascript
      const p = document.getElementById("c-promoted");    if (p) p.textContent = summary.promoted;
      const sc = document.getElementById("c-screened");    if (sc) sc.textContent = summary.screened;
      const s = document.getElementById("c-spend");       if (s) s.textContent = "$" + Number(summary.cumulative_spend_usd).toFixed(2);
```

- [ ] **Step 4: Explain a skipped WFO section in `detail.html`**

In `factory/dashboard/templates/detail.html`, the WFO section (`<h2>Stage 3 — WFO</h2>`) ends with this line (uniquely identified by the preceding `record.wfo.run_bundle_path` line):

```html
      <li><strong>run_bundle_path:</strong> {{ record.wfo.run_bundle_path }}</li>
    </ul>
  {% else %}<p>(not reached)</p>{% endif %}
```

Change the final line to add a screened-out branch:

```html
      <li><strong>run_bundle_path:</strong> {{ record.wfo.run_bundle_path }}</li>
    </ul>
  {% elif record.screened_out %}<p>(WFO skipped &mdash; screened out: {{ record.screen_reason }})</p>{% else %}<p>(not reached)</p>{% endif %}
```

Leave the backtest and optimize `{% else %}<p>(not reached)</p>{% endif %}` lines unchanged — only the WFO one changes.

- [ ] **Step 5: Verify the dashboard module imports cleanly**

Run: `python -c "import factory.dashboard.server"`
Expected: no output, exit 0.

- [ ] **Step 6: Re-run the full non-slow suite**

Run: `python -m pytest factory/tests -q -m "not slow"`
Expected: all tests still pass (dashboard changes do not affect the cycle/results tests).

---

### Task 6: Commit

- [ ] **Step 1: Stage exactly the touched files and commit**

```bash
git add factory/config/settings.toml factory/settings_loader.py \
        factory/results.py factory/cycle.py factory/dashboard/ \
        factory/tests/conftest.py factory/tests/test_cycle.py \
        docs/superpowers/plans/2026-05-15-screen-out-before-wfo.md
git commit -m "$(cat <<'EOF'
feat(factory): screen out hopeless strategies before WFO (Task A)

After the optimize stage, if best in-sample score is below
settings.screening.min_optimize_score (default 1.3), skip the WFO stage.
OOS Sharpe is almost always below the best in-sample score, so a hopeless
optimize means WFO won't shortlist it -- skipping the most expensive stage
saves wall-clock with near-zero false-negative risk. Screened cycles record
status=complete with screened_out=true, wfo=null, promotion=null. Dashboard
shows a Screened counter and a 'complete (screened)' status.
EOF
)"
```

Note: `git add factory/dashboard/` stages all four dashboard files. If `factory/dashboard/` contains other unrelated modifications, stage the four files individually instead. Do **not** stage anything under `backtester/`. Do **not** push.

- [ ] **Step 2: Confirm the commit**

Run: `git status` and `git log -1 --stat`
Expected: a single new commit; only the files listed above appear; nothing under `backtester/` is staged or committed.

---

## Self-Review

**Spec coverage:**
- Change 1 (settings) → Task 2 ✓
- Change 2 (results schema) → Task 3 ✓
- Change 3 (cycle.py gate) → Task 4 ✓
- Change 4 (dashboard) → Task 5 ✓
- Change 5 (tests) → Task 1 ✓
- Commit → Task 6 ✓

**Spec checklist answered:**
- `ScreeningCfg` defaults to `enabled=False` when `[screening]` absent — yes, `sc.get("enabled", False)` (Task 2 Step 4).
- Screen fires only between optimize and wfo, only when enabled — yes, `if stage_name == "wfo" and s.screening.enabled` (Task 4 Step 2).
- A screened cycle: `status=complete`, `wfo=None`, `promotion=None`, `screened_out=true` — yes (Task 4 Steps 4–6; `CycleOutcome` status unchanged).
- Promotion block no longer dereferences `wfo.parsed` when `wfo is None` — yes, `and wfo is not None` guard (Task 4 Step 4).
- `build_record` / `build_failed_record` both carry `screened_out` + `screen_reason` — yes (Task 3 Steps 1–2).
- New test asserts `run_wfo_stage` not called — yes, `run_wfo_stage.assert_not_called()` (Task 1 Step 2).
- Full not-slow suite green — verified at Task 4 Step 8 and Task 5 Step 6.

**Placeholder scan:** none — every code step shows complete content.

**Type consistency:** `ScreeningCfg(enabled: bool, min_optimize_score: float)` defined in Task 2 and used as `s.screening.enabled` / `s.screening.min_optimize_score` in Task 4. `screened_out` / `screen_reason` keys are produced in Task 3 and consumed identically (`r.get("screened_out")`, `record.screen_reason`) in Tasks 4–5. Consistent throughout.

## Report

After Task 6, report:
- Status: `DONE` | `DONE_WITH_CONCERNS` | `BLOCKED`
- Full `python -m pytest factory/tests -q -m "not slow"` output
- Commit SHA
- Any self-review findings
