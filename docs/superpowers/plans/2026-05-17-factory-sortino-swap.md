# Factory Sortino Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the strategy factory to select, alert on, and promote generated strategies using out-of-sample Sortino instead of Sharpe, end to end.

**Architecture:** The WFO summary parser starts emitting `oos_sortino` alongside the retained `oos_sharpe`. The promotion stage, alert gate, and in-sample optimisation objective all move to Sortino via a config-key rename plus targeted code edits. `oos_sharpe` stays in the record as a display-only field. Dashboard read-sites that depend on renamed promotion fields are updated so the page does not break; the broader dashboard metric rework is out of scope.

**Tech Stack:** Python 3.11, pytest, TOML config, Jinja2 templates, vanilla JS.

**Reference spec:** `docs/superpowers/specs/2026-05-17-factory-sortino-swap-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `factory/stages.py` | WFO summary parsing | Emit `oos_sortino` |
| `factory/settings_loader.py` | Config dataclasses + loader | Rename `PromotionCfg.min_avg_sharpe`, update `trigger_metric` default |
| `factory/promote.py` | Held-out promotion stage | Aggregate Sortino, rename `PromotionResult` fields |
| `factory/cycle.py` | Cycle orchestration | Log line + stale comment |
| `factory/notify.py` | Telegram alert formatting | Alert body headlines Sortino |
| `factory/prompt.py` | Generation prompt template | In-sample objective → `sortino` |
| `factory/config/settings.toml` | Production config | Flip alert + promotion metrics |
| `factory/dashboard/templates/detail.html` | Strategy detail page | Renamed promotion field read-sites |
| `factory/dashboard/templates/overview.html` | Overview table | Renamed promotion field read-site |
| `factory/dashboard/static/overview.js` | Overview table JS | Renamed promotion field read-sites |
| `factory/tests/fixtures/sample_wfo_summary.json` | Parser test fixture | Add `sortino` key |
| `factory/tests/test_stages_parsers.py` | Parser tests | Assert `oos_sortino` |
| `factory/tests/test_promote.py` | Promotion tests | Sortino aggregation + renamed fields |
| `factory/tests/test_notify.py` | Alert tests | Sortino in record + alert body |

**Task order matters:** Task 1 → Task 2 → Task 3 (each depends on the previous). Tasks 4, 5, 6 are independent and may run in any order after Task 1.

---

### Task 1: WFO parser emits `oos_sortino`

**Files:**
- Modify: `factory/tests/fixtures/sample_wfo_summary.json`
- Modify: `factory/stages.py:75-86` (`parse_wfo_summary`)
- Test: `factory/tests/test_stages_parsers.py:43-51`
- Modify: `factory/tests/test_promote.py:69-82` (`_seed_promotion_bundle` — keep it green)

- [ ] **Step 1: Add `sortino` to the parser test fixture**

Overwrite `factory/tests/fixtures/sample_wfo_summary.json` with:

```json
{
  "oos_summary": {
    "total_return": 0.31078674058870415,
    "sharpe": 0.6897185480952924,
    "sortino": 0.9847132205534219,
    "max_drawdown": -0.07374129512318983,
    "n_trades": 109,
    "win_rate": 0.660377358490566
  },
  "is_summary_avg": {"sharpe": 1.0344, "total_return": 0.1859},
  "parameter_stability": {
    "entry_percentile": {"unique": 2, "mode": 30.0, "values_by_window": [30.0, 30.0, 30.0, 10.0, 10.0, 10.0]},
    "max_hold": {"unique": 2, "mode": 10, "values_by_window": [10, 10, 10, 5, 5, 5]}
  },
  "n_windows": 6
}
```

- [ ] **Step 2: Write the failing assertion**

In `factory/tests/test_stages_parsers.py`, in `test_parse_wfo_summary_extracts_oos_block`, add the `oos_sortino` assertion right after the `oos_sharpe` one:

```python
def test_parse_wfo_summary_extracts_oos_block() -> None:
    raw = _load("sample_wfo_summary.json")
    parsed = parse_wfo_summary(raw, bundle_path=Path("output/runs/z"))
    assert parsed["oos_sharpe"] == pytest.approx(0.6897185480952924)
    assert parsed["oos_sortino"] == pytest.approx(0.9847132205534219)
    assert parsed["oos_total_return"] == pytest.approx(0.31078674058870415)
    assert parsed["oos_max_drawdown"] == pytest.approx(-0.07374129512318983)
    assert parsed["oos_n_trades"] == 109
    assert parsed["n_windows"] == 6
    assert parsed["parameter_stability"]["entry_percentile"]["mode"] == 30.0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest factory/tests/test_stages_parsers.py::test_parse_wfo_summary_extracts_oos_block -v`
Expected: FAIL with `KeyError: 'oos_sortino'`.

- [ ] **Step 4: Add `oos_sortino` to the parser**

In `factory/stages.py`, in `parse_wfo_summary`, add the `oos_sortino` line. The function becomes:

```python
def parse_wfo_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map WFO summary.json (nested oos_summary) onto the factory record shape."""
    oos = raw["oos_summary"]
    return {
        "oos_sharpe": float(oos["sharpe"]),
        "oos_sortino": float(oos["sortino"]),
        "oos_total_return": float(oos["total_return"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_n_trades": int(oos["n_trades"]),
        "parameter_stability": dict(raw.get("parameter_stability", {})),
        "n_windows": int(raw["n_windows"]),
        "run_bundle_path": bundle_path.as_posix(),
    }
```

- [ ] **Step 5: Keep the promotion test fixture parseable**

`parse_wfo_summary` now requires a `sortino` key. The promotion tests seed their own `summary.json` files via `_seed_promotion_bundle`, which currently omits `sortino`. In `factory/tests/test_promote.py`, add a `sortino` entry to that helper's `oos_summary` (reuse the existing argument value — Task 2 renames the argument properly):

```python
def _seed_promotion_bundle(output_runs_dir: Path, run_name: str, oos_sharpe: float) -> None:
    """Pre-create a bundle dir + summary.json so the parser can find it."""
    bundle = output_runs_dir / f"20260101_0900_{run_name}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps({
        "oos_summary": {
            "sharpe": oos_sharpe,
            "sortino": oos_sharpe,
            "total_return": 0.18,
            "max_drawdown": -0.07,
            "n_trades": 80,
        },
        "parameter_stability": {},
        "n_windows": 6,
    }), encoding="utf-8")
```

- [ ] **Step 6: Run the parser and promotion tests to verify they pass**

Run: `python -m pytest factory/tests/test_stages_parsers.py factory/tests/test_promote.py -v`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add factory/stages.py factory/tests/fixtures/sample_wfo_summary.json factory/tests/test_stages_parsers.py factory/tests/test_promote.py
git commit -m "feat: parse oos_sortino from WFO summary into factory record"
```

---

### Task 2: Promotion stage decides on Sortino

This task renames `PromotionCfg.min_avg_sharpe` → `min_avg_sortino` and `PromotionResult.avg_sharpe`/`min_avg_sharpe_threshold` → `avg_sortino`/`min_avg_sortino_threshold`. The rename ripples through the loader, `promote.py`, `cycle.py`, the dashboard read-sites, and the promotion tests — it must land as one atomic commit or intermediate states fail.

**Files:**
- Modify: `factory/settings_loader.py:66` and `:173-174`
- Modify: `factory/promote.py` (`PromotionResult` dataclass, docstrings, `promote_strategy`)
- Modify: `factory/cycle.py:234-238`
- Modify: `factory/dashboard/templates/detail.html:99-101`
- Modify: `factory/dashboard/templates/overview.html:61`
- Modify: `factory/dashboard/static/overview.js:34` and `:97`
- Test: `factory/tests/test_promote.py`

- [ ] **Step 1: Update the promotion tests to expect Sortino**

In `factory/tests/test_promote.py`, replace the `_cfg` helper and `_seed_promotion_bundle` helper:

```python
def _cfg(min_avg_sortino: float = 0.7, tickers: tuple[str, ...] = ("AAPL", "QQQ", "DIA")) -> PromotionCfg:
    return PromotionCfg(
        enabled=True,
        tickers=tickers,
        data_source="yfinance",
        min_avg_sortino=min_avg_sortino,
        trigger_metric="wfo.oos_sortino",
        trigger_threshold=1.0,
    )
```

```python
def _seed_promotion_bundle(output_runs_dir: Path, run_name: str, oos_sortino: float) -> None:
    """Pre-create a bundle dir + summary.json so the parser can find it."""
    bundle = output_runs_dir / f"20260101_0900_{run_name}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps({
        "oos_summary": {
            "sharpe": oos_sortino,
            "sortino": oos_sortino,
            "total_return": 0.18,
            "max_drawdown": -0.07,
            "n_trades": 80,
        },
        "parameter_stability": {},
        "n_windows": 6,
    }), encoding="utf-8")
```

Then update every caller and assertion in that file:
- `test_promote_strategy_passes_when_all_tickers_clear_avg_threshold`: change `promotion_cfg=_cfg(min_avg_sharpe=0.7)` → `promotion_cfg=_cfg(min_avg_sortino=0.7)`; change `assert result.avg_sharpe == pytest.approx(0.8)` → `assert result.avg_sortino == pytest.approx(0.8)`; change `assert result.per_ticker["AAPL"]["oos_sharpe"] == pytest.approx(0.9)` → `assert result.per_ticker["AAPL"]["oos_sortino"] == pytest.approx(0.9)`.
- `test_promote_strategy_fails_when_avg_below_threshold`: change `promotion_cfg=_cfg(min_avg_sharpe=0.7)` → `promotion_cfg=_cfg(min_avg_sortino=0.7)`; change `assert result.avg_sharpe == pytest.approx(0.4)` → `assert result.avg_sortino == pytest.approx(0.4)`.
- `test_promote_strategy_continues_on_per_ticker_failure`: change `promotion_cfg=_cfg(min_avg_sharpe=0.5)` → `promotion_cfg=_cfg(min_avg_sortino=0.5)`; change `assert result.avg_sharpe == pytest.approx(0.9)` → `assert result.avg_sortino == pytest.approx(0.9)`.
- `test_promote_strategy_fails_when_subprocess_nonzero`: change `assert result.avg_sharpe is None` → `assert result.avg_sortino is None`.

- [ ] **Step 2: Run the promotion tests to verify they fail**

Run: `python -m pytest factory/tests/test_promote.py -v`
Expected: FAIL — `PromotionCfg.__init__` rejects `min_avg_sortino` (unexpected keyword), and `PromotionResult` has no `avg_sortino`.

- [ ] **Step 3: Rename the config field in `settings_loader.py`**

In `factory/settings_loader.py`, the `PromotionCfg` dataclass — change line 66:

```python
@dataclass(slots=True, frozen=True)
class PromotionCfg:
    enabled: bool
    tickers: tuple[str, ...]
    data_source: str
    min_avg_sortino: float
    trigger_metric: str
    trigger_threshold: float
```

And the loader block (currently lines 169-176) — change the `min_avg_sortino` line and the `trigger_metric` default:

```python
        promotion=PromotionCfg(
            enabled=bool(pr.get("enabled", False)),
            tickers=tuple(pr.get("tickers", ())),
            data_source=str(pr.get("data_source", "yfinance")),
            min_avg_sortino=float(pr.get("min_avg_sortino", 0.7)),
            trigger_metric=str(pr.get("trigger_metric", "wfo.oos_sortino")),
            trigger_threshold=float(pr.get("trigger_threshold", 1.0)),
        ),
```

- [ ] **Step 4: Switch `promote.py` to Sortino**

In `factory/promote.py`:

Module docstring (lines 6-7) — change:
```python
suffixed) and run a full WFO via subprocess. Aggregate OOS Sortino across the
panel; gate against min_avg_sortino.
```

`PromotionResult` dataclass (lines 36-44) — change the two field names:
```python
@dataclass(slots=True, frozen=True)
class PromotionResult:
    ran: bool
    tickers: tuple[str, ...]
    per_ticker: dict[str, dict[str, Any]]
    avg_sortino: Optional[float]
    min_avg_sortino_threshold: float
    passed: bool
    error: Optional[str] = None
```

`promote_strategy` docstring (lines 135-136) — change:
```python
    data). passed=True only if ALL tickers succeed AND avg oos_sortino clears
    promotion_cfg.min_avg_sortino.
```

Per-ticker log line (lines 158-161) — change:
```python
            per_ticker[ticker] = parsed
            log.info(
                "promotion %s ticker=%s oos_sortino=%.3f",
                strategy_id, ticker, parsed.get("oos_sortino", 0.0),
            )
```

Aggregation block (lines 166-177) — change:
```python
    if per_ticker:
        sortinos = [float(p["oos_sortino"]) for p in per_ticker.values()]
        avg = sum(sortinos) / len(sortinos)
    else:
        avg = None

    all_tickers_completed = len(per_ticker) == len(promotion_cfg.tickers)
    passed = (
        all_tickers_completed
        and avg is not None
        and avg >= promotion_cfg.min_avg_sortino
    )
```

Return statement (lines 179-187) — change the two renamed fields:
```python
    return PromotionResult(
        ran=True,
        tickers=tuple(promotion_cfg.tickers),
        per_ticker=per_ticker,
        avg_sortino=avg,
        min_avg_sortino_threshold=promotion_cfg.min_avg_sortino,
        passed=passed,
        error=error,
    )
```

- [ ] **Step 5: Update the promotion log line in `cycle.py`**

In `factory/cycle.py`, the promotion-result log (lines 234-238) — change `avg_sharpe` to `avg_sortino`:

```python
            log.info(
                "cycle id=%s promotion passed=%s avg_sortino=%s",
                strategy_id, promo.passed,
                f"{promo.avg_sortino:.3f}" if promo.avg_sortino is not None else "n/a",
            )
```

- [ ] **Step 6: Update the dashboard read-sites for the renamed fields**

In `factory/dashboard/templates/detail.html`, lines 99-101 — change to:
```html
      &nbsp;(avg OOS Sortino across tickers
      <strong>{{ "%.3f" | format(record.promotion.avg_sortino) if record.promotion.avg_sortino is not none else 'n/a' }}</strong>
      vs threshold {{ "%.2f" | format(record.promotion.min_avg_sortino_threshold) }})
```

In `factory/dashboard/templates/overview.html`, line 61 — change to:
```html
        <td>{{ "%.3f"|format(r.promotion.avg_sortino) if r.promotion and r.promotion.avg_sortino is not none else '' }}</td>
```

In `factory/dashboard/static/overview.js`, line 34 — change to:
```javascript
    const promoAvg = (promo && typeof promo.avg_sortino === "number") ? fmt(promo.avg_sortino) : "";
```

In `factory/dashboard/static/overview.js`, line 97 — change to:
```javascript
    if (key === "promotion_avg_sharpe") return rec.promotion ? rec.promotion.avg_sortino : null;
```

(The `data-sort="promotion_avg_sharpe"` attribute string in `overview.html:43` is an opaque sort key — leave it unchanged; only the property access changes.)

- [ ] **Step 7: Run the promotion tests to verify they pass**

Run: `python -m pytest factory/tests/test_promote.py -v`
Expected: PASS (all tests).

- [ ] **Step 8: Run the full factory suite**

Run: `python -m pytest factory/tests/ -q`
Expected: PASS — no regressions.

- [ ] **Step 9: Commit**

```bash
git add factory/settings_loader.py factory/promote.py factory/cycle.py factory/dashboard/templates/detail.html factory/dashboard/templates/overview.html factory/dashboard/static/overview.js factory/tests/test_promote.py
git commit -m "feat: promotion stage gates on OOS Sortino instead of Sharpe"
```

---

### Task 3: Flip production config to Sortino

**Files:**
- Modify: `factory/config/settings.toml`

- [ ] **Step 1: Update the alert metric**

In `factory/config/settings.toml`, in the `[alerts]` section, change:
```toml
alert_threshold_metric = "wfo.oos_sortino"
```
(`alert_threshold` stays `1.0`.)

- [ ] **Step 2: Update the promotion metric and threshold key**

In `factory/config/settings.toml`, in the `[promotion]` section, change `min_avg_sharpe` to `min_avg_sortino` and `trigger_metric` to the Sortino path. The section becomes:
```toml
[promotion]
enabled           = true
tickers           = ["AAPL", "QQQ", "DIA"]
data_source       = "yfinance"        # csv if all tickers in data/raw/; yfinance fetches on first run
min_avg_sortino   = 0.7
trigger_metric    = "wfo.oos_sortino"
trigger_threshold = 1.0
```

- [ ] **Step 3: Verify the config still loads**

Run: `python -c "from pathlib import Path; from factory.settings_loader import load_settings; s = load_settings(Path('factory/config/settings.toml')); print(s.alerts.alert_threshold_metric, s.promotion.trigger_metric, s.promotion.min_avg_sortino)"`
Expected output: `wfo.oos_sortino wfo.oos_sortino 0.7`

- [ ] **Step 4: Run the full factory suite**

Run: `python -m pytest factory/tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add factory/config/settings.toml
git commit -m "feat: point factory alert + promotion metrics at OOS Sortino"
```

---

### Task 4: Alert message headlines OOS Sortino

**Files:**
- Modify: `factory/notify.py:15` and `:52-60`
- Test: `factory/tests/test_notify.py`

- [ ] **Step 1: Update the alert test record and assertions**

In `factory/tests/test_notify.py`, update the `_record` helper to carry `oos_sortino`:

```python
def _record() -> dict:
    return {
        "strategy_id": "gen_42",
        "status": "complete",
        "idea": {"one_line_summary": "compression breakout test"},
        "backtest": {"sharpe": 0.9},
        "optimize": {"best_score": 1.4},
        "wfo": {"oos_sharpe": 1.25, "oos_sortino": 1.40, "oos_total_return": 0.18,
                "oos_max_drawdown": -0.06, "oos_n_trades": 25},
    }
```

In `test_format_alert_message_labels_as_shortlist_signal`, change the metric assertion to check the Sortino headline:
```python
    assert "1.40" in msg or "1.400" in msg  # oos_sortino
```

In all four `maybe_send_alert` tests (`test_maybe_send_alert_skips_when_metric_below_threshold`, `test_maybe_send_alert_skips_when_credentials_missing`, `test_maybe_send_alert_calls_telegram_when_above_threshold`, `test_maybe_send_alert_swallows_telegram_failures`), change `alert_threshold_metric="wfo.oos_sharpe"` to `alert_threshold_metric="wfo.oos_sortino"`. The thresholds already work against the record's `oos_sortino` of `1.40`: the `2.0` threshold still skips (below), the `1.0` thresholds still send (above). Also update the inline comment on the `2.0` threshold in `test_maybe_send_alert_skips_when_metric_below_threshold` from `# above this record's 1.25` to `# above this record's 1.40`.

- [ ] **Step 2: Run the alert tests to verify they fail**

Run: `python -m pytest factory/tests/test_notify.py -v`
Expected: FAIL — `test_format_alert_message_labels_as_shortlist_signal` fails (message has no `1.40`).

- [ ] **Step 3: Update the alert message and the config comment**

In `factory/notify.py`, line 15 — update the example comment:
```python
    alert_threshold_metric: str  # e.g. "wfo.oos_sortino"
```

In `format_alert_message` (lines 52-60), add the OOS Sortino line as the headline metric, above the retained OOS Sharpe line:
```python
    wfo = record.get("wfo") or {}
    parts = [
        "[SHORTLIST SIGNAL — not a verdict]",
        f"Strategy: {sid}",
        f"Idea: {summary}",
        f"OOS Sortino: {wfo.get('oos_sortino', 'n/a')}",
        f"OOS Sharpe: {wfo.get('oos_sharpe', 'n/a')}",
        f"OOS total return: {wfo.get('oos_total_return', 'n/a')}",
        f"OOS max drawdown: {wfo.get('oos_max_drawdown', 'n/a')}",
        f"OOS trades: {wfo.get('oos_n_trades', 'n/a')}",
        "",
        "This cleared the configured threshold metric on a single historical",
        "path. A held-out gate (different symbol or fully unseen period) is",
        "required before treating this as a real candidate.",
        "",
        f"Detail: {dashboard_base_url.rstrip('/')}/strategy/{sid}",
    ]
    return "\n".join(parts)
```

- [ ] **Step 4: Run the alert tests to verify they pass**

Run: `python -m pytest factory/tests/test_notify.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add factory/notify.py factory/tests/test_notify.py
git commit -m "feat: alert message headlines OOS Sortino"
```

---

### Task 5: In-sample WFO objective becomes Sortino

**Files:**
- Modify: `factory/prompt.py:128`

- [ ] **Step 1: Change the objective in the prompt config template**

In `factory/prompt.py`, in the embedded config template, change line 128 from:
```
  objective: sharpe
```
to:
```
  objective: sortino
```

- [ ] **Step 2: Run the prompt tests and the full factory suite**

Run: `python -m pytest factory/tests/test_prompt.py factory/tests/ -q`
Expected: PASS — `test_prompt.py` asserts nothing about the objective, so this is a clean change.

- [ ] **Step 3: Commit**

```bash
git add factory/prompt.py
git commit -m "feat: generated WFO configs optimize in-sample on sortino"
```

---

### Task 6: Update the stale comment and completion log in `cycle.py`

**Files:**
- Modify: `factory/cycle.py:252-254` and `:259-262`

- [ ] **Step 1: Update the stale alert-trigger comment**

In `factory/cycle.py`, the comment block before `maybe_send_alert` (lines 252-254) — change:
```python
    # Step 16: alert (conditional). maybe_send_alert never raises.
    # NOTE: alert trigger fires on the configured alert_threshold_metric
    # (wfo.oos_sortino — see factory/config/settings.toml).
    # Promotion is informational on the dashboard, not a gate on alerts.
```

- [ ] **Step 2: Update the cycle completion log line**

In `factory/cycle.py`, the completion log (lines 259-262) — change `oos_sharpe` to `oos_sortino`:
```python
    log.info("cycle id=%s complete oos_sortino=%s screened=%s alerted=%s",
             strategy_id,
             wfo.parsed.get("oos_sortino") if wfo is not None else "n/a",
             screened_out, rec["alerted"])
```

- [ ] **Step 3: Run the cycle tests and the full factory suite**

Run: `python -m pytest factory/tests/test_cycle.py factory/tests/ -q`
Expected: PASS — `test_cycle.py` does not assert on log output, and promotion is disabled in the test fixture.

- [ ] **Step 4: Commit**

```bash
git add factory/cycle.py
git commit -m "refactor: cycle log + comment reference oos_sortino"
```

---

## Final Verification

- [ ] **Run the full factory test suite one last time**

Run: `python -m pytest factory/tests/ -q`
Expected: PASS — all tests green.

- [ ] **Confirm `oos_sharpe` is display-only**

Grep for `oos_sharpe` across `factory/` and confirm every remaining hit is either: the parser writing it (`stages.py`), a log line (`cycle.py`), the alert body (`notify.py`), a dashboard display/sort site (`dashboard/`), or a test. No gate, screening, or promotion logic should reference it.

Run: `git grep -n "oos_sharpe" factory/`

---

## Out of Scope (per spec)

- Hourly timeframe.
- Drawdown veto gate.
- Broader dashboard rework: re-labelling the `OOS Sharpe` column, adding an `OOS Sortino` column, the `THRESHOLD_METRIC` default in `overview.js:3`. The dashboard keeps showing `OOS Sharpe` (still a valid retained field); only the renamed promotion fields are updated here to avoid breaking the page.
- `backtester/wfo/multi_stitcher.py` — its hand-built `oos_summary` omits `sortino`; not exercised by the single-symbol factory today.
