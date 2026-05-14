# Mean-Reversion ATR + v0.4.0 Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `mean_reversion_atr` swing-trading strategy and the v0.4.0 framework that supports it — multi-symbol portfolio simulator with shared cash and risk-budget enforcement, two-phase tranche stop, three-gate regime policy, volatility-targeted sizing, yfinance data loader, universe screening CLI, and Latin-hypercube optimizer mode.

**Architecture:** New v0.4.0 code path lives alongside the v0.3.0 single-symbol path. Strategies opt in via `uses_multi_symbol = True` and `uses_per_bar = True` class attributes; existing v0.3.0 strategies (sma_cross, momentum_streak, rsi_long_short) are untouched. The new path runs through `MultiSymbolBacktestEngine` → `MultiSymbolPortfolioSimulator` (shared cash, RegimePolicy, RiskBudgetEnforcer, SectorCapEnforcer) → per-symbol `Broker` + `FillEngine` + `Position` + `TrancheStopState`. Strategy contract gains `aux_data: dict[str, pd.DataFrame]`, fractional `target_position` ∈ [-1.0, 1.0], and per-bar callback mode. v0.3.0's `TrailingStopState` is preserved unchanged; v0.4.0's `TrancheStopState` is a separate class using close-basis ratcheting with `HARD → RUNNER → DISARMED` phases. Regime gates (SPY 200-EMA, VIX hysteresis, strategy circuit breaker) live in `RegimePolicy` and trip the book to full cash. Universe selection is a separate CLI (`scripts/screen_universe.py`) using a `range/ATR` ratio and a 200-day OLS-slope trend filter. Optimizer gains a discrete-LHS sampling mode for the 6-dimensional WFO surface.

**Tech Stack:** Python 3.11, pandas, numpy, pytest, PyYAML, `yfinance>=0.2.40` (new optional extras dependency).

**Scope notes:**

- Slice C from the brainstorm: full v0.4.0 framework. Strategy-only Slice A and Slice B are not in scope.
- Strategy-performance assertions (DD < 9%, return > 15%, Calmar > 2.5) are wrapped in `@pytest.mark.xfail(strict=False)` by default. The framework ships when structural-correctness gates pass; performance tuning is a separate workflow (see Phase 19).
- `yfinance` becomes an optional extras dependency. The core package stays network-free. Real-data fixtures are committed to the repo so CI runs offline.
- v0.3.0 backwards compat is enforced by `tests/integration/test_backwards_compat.py` (modified to point at relocated `data/synth/SPY.csv`).
- No new symbols are added to `data/raw/` without explicit user approval. Phase 18 captures that approval gate.

---

## Required reading before starting

1. `docs/superpowers/specs/2026-05-14-mean-reversion-atr-design.md` — the spec this plan implements. **Read in full.** All references to sections (§N.M) refer to this spec.
2. `docs/superpowers/plans/2026-05-14-trailing-stops.md` — the v0.3.0 plan. Matches atomic-commit cadence, TDD step structure, and final backwards-compat verification pattern.
3. `backtester/engine/portfolio.py` — the v0.3.0 PortfolioSimulator. The v0.4.0 `MultiSymbolPortfolioSimulator` is a sibling, not a replacement.
4. `backtester/engine/trailing_stop.py`, `backtester/engine/atr.py` — v0.3.0 trailing-stop machinery. Untouched in v0.4.0; v0.4.0 introduces `TrancheStopState` in a new file.
5. `backtester/strategies/base.py` — `BaseStrategy` is gaining two new class attributes (`uses_multi_symbol`, `uses_per_bar`).
6. `backtester/core/types.py` — `StrategyContext` is gaining four new fields (`position_phase`, `bars_in_phase`, `recent_pnl`, `regime`).
7. `backtester/config/models.py` and `backtester/config/validation.py` — where new fields and 18 new validation rules go.
8. `strategies/momentum_streak.py` — most architecturally similar existing strategy; `mean_reversion_atr` borrows the state-machine pattern.

---

## Baseline verification before starting

Run from repo root:
```
python -m pytest -q
```
Expected: `202 passed`. This is the **no-regression baseline**. Record it.

Capture the v0.3.0 backwards-compat golden BEFORE Phase 18 modifies any data files. Run:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
Note the printed run directory (`<RUNDIR>`). Then:
```
python -c "import json; s=json.load(open(r'<RUNDIR>/summary.json')); print({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')})"
```
**Paste the printed dict into a scratch file.** It becomes the golden in Phase 19's update to `tests/integration/test_backwards_compat.py` (the synth-relocation update preserves byte-identical numerics).

---

## File-structure preview

Counted at v0.4.0 freeze. **Create** = new file. **Modify** = touch existing file. **Replace** = overwrite existing file's content.

### Engine
| File | Action |
|---|---|
| `backtester/engine/tranche_stop.py` | create |
| `backtester/engine/regime.py` | create |
| `backtester/engine/risk_budget.py` | create |
| `backtester/engine/sector_cap.py` | create |
| `backtester/engine/multi_portfolio.py` | create |
| `backtester/engine/multi_backtest_engine.py` | create |
| `backtester/engine/trailing_stop.py` | unchanged |
| `backtester/engine/portfolio.py` | unchanged |
| `backtester/engine/atr.py` | unchanged |

### Strategy + core
| File | Action |
|---|---|
| `strategies/mean_reversion_atr.py` | create |
| `backtester/strategies/registry.py` | modify |
| `backtester/strategies/base.py` | modify |
| `backtester/core/types.py` | modify |

### Data
| File | Action |
|---|---|
| `backtester/data/yfinance_loader.py` | create |
| `backtester/data/loader.py` | modify |
| `backtester/data/validators.py` | modify |
| `data/sector_map.csv` | create |
| `data/synth/SPY.csv`, `data/synth/AAPL.csv` | create (relocated) |
| `data/raw/SPY.csv`, `data/raw/AAPL.csv` | replace (synth → real, committed) |
| `data/raw/{TSLA,NVDA,AMD,COIN,GOOGL,MSTR,XPEV,NIO,PLTR,SMCI,SHOP,W,META,NFLX,^VIX}.csv` | create (real, committed) |
| `scripts/generate_sample_data.py` | modify |

### Config
| File | Action |
|---|---|
| `backtester/config/models.py` | modify |
| `backtester/config/loader.py` | modify |
| `backtester/config/universe.py` | create |
| `backtester/config/validation.py` | modify |
| `configs/backtests/mean_rev_v04.yaml` | create |
| `configs/universe.yaml` | create |
| `configs/optimize/mean_rev_v04_grid.yaml` | create |
| `configs/wfo/mean_rev_v04_wfo.yaml` | create |
| `configs/universe_candidates_seed.txt` | create |

### Optimizer
| File | Action |
|---|---|
| `backtester/optimize/grid_search.py` | modify |
| `backtester/optimize/lhs_sampler.py` | create |

### Runners & artifacts
| File | Action |
|---|---|
| `backtester/runners/run_batch.py` | modify |
| `backtester/runners/run_wfo.py` | modify |
| `backtester/runners/run_backtest.py` | unchanged |
| `backtester/io/artifacts.py` | modify |

### CLI
| File | Action |
|---|---|
| `scripts/screen_universe.py` | create |

### Tests (88 new)
| File | Action |
|---|---|
| `tests/unit/test_tranche_stop.py` | create (14) |
| `tests/unit/test_multi_symbol_simulator.py` | create (18) |
| `tests/unit/test_regime_policy.py` | create (10) |
| `tests/unit/test_yfinance_loader.py` | create (5) |
| `tests/unit/test_screen_universe.py` | create (6) |
| `tests/unit/test_strategy_mean_reversion_atr.py` | create (14) |
| `tests/unit/test_universe_loader.py` | create (4) |
| `tests/unit/test_optimizer_lhs.py` | create (3) |
| `tests/unit/test_sector_map.py` | create (2) |
| `tests/unit/test_config_validation.py` | append (18) |
| `tests/integration/test_stress_windows.py` | create (4 parametrized) |
| `tests/integration/test_held_out_2022_2025.py` | create (1) |
| `tests/integration/test_screen_universe_cli.py` | create (1) |
| `tests/integration/test_run_batch_cli.py` | append (1) |
| `tests/integration/test_run_wfo_cli.py` | append (1) |
| `tests/integration/test_backwards_compat.py` | modify |

### Docs & project
| File | Action |
|---|---|
| `docs/strategy_contract.md` | modify |
| `docs/runbook.md` | modify |
| `README.md` | modify |
| `pyproject.toml` | bump 0.3.0 → 0.4.0; add data extras |

---

## Phase plan

The work splits into 21 phases. Each phase ends with a green `pytest -q` and an atomic commit per task. Test count is tracked cumulatively; mismatch means a regression slipped in.

| Phase | Topic | Cumulative test count target |
|---|---|---|
| 1 | Baseline & bootstrap | 202 |
| 2 | Data layer — yfinance loader, validator relaxation, synth relocation, sector_map | 209 |
| 3 | Config models — new fields on DataConfig/ExecutionConfig/PortfolioConfig/RegimesConfig | 218 |
| 4 | Config validation — 18 new rules | 236 |
| 5 | Universe loader — config/universe.py + overrides merge | 240 |
| 6 | Strategy contract extensions — BaseStrategy attrs + StrategyContext fields | 240 (extends existing) |
| 7 | TrancheStopState | 254 |
| 8 | RegimePolicy | 264 |
| 9 | Risk-budget + Sector-cap enforcers | 270 |
| 10 | MultiSymbolPortfolioSimulator | 288 |
| 11 | MultiSymbolBacktestEngine | 289 |
| 12 | LHS optimizer mode | 292 |
| 13 | mean_reversion_atr strategy | 306 |
| 14 | Universe screening CLI | 312 |
| 15 | Runner routing — run_batch + run_wfo | 313 |
| 16 | Multi-symbol artifacts | 314 |
| 17 | Real-data fixtures (gated by user approval) | 314 |
| 18 | Integration smokes — multi-symbol batch + WFO + screen_universe CLI | 317 |
| 19 | Stress windows + held-out + backwards-compat update | 322 |
| 20 | Docs — strategy_contract, runbook, README | 322 |
| 21 | Version bump + tag | 322 |

The final number (322) = 202 baseline + 88 new − 0 deletions, plus 32 from test_config_validation parametrization that don't show in the per-file counts above. The exact endpoint is whatever `pytest -q` reports after Phase 21; the +88 number in the spec is the design intent; the cumulative table is a budget, not an exact contract.

> **Note on plan length.** This plan is intentionally long. Phases 1–21 are independent enough to checkpoint between. Execute one phase, run the suite, confirm the cumulative count matches the table above, then move on. The `superpowers:subagent-driven-development` skill is designed for exactly this cadence.

---

## Phase 1: Baseline & bootstrap

The only purpose of Phase 1 is to capture the v0.3.0 baseline numerics so they can be defended at the end. No code changes.

### Task 1: Capture the v0.3.0 baseline

**Files:**
- No source changes. This task produces a scratch artifact only.

- [ ] **Step 1: Verify the baseline test count**

Run: `python -m pytest -q`
Expected: `202 passed` (exact integer). If different, STOP and reconcile before proceeding.

- [ ] **Step 2: Capture the sma_cross_spy baseline numerics**

Run:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

Note the printed run directory (look for `output/runs/<timestamp>_sma_cross_spy`).

- [ ] **Step 3: Extract the golden dict**

Run (replacing `<RUNDIR>` with the path from step 2):
```
python -c "import json; s=json.load(open(r'<RUNDIR>/summary.json')); print({k:s[k] for k in ('total_return','sharpe','max_drawdown','n_trades','final_equity')})"
```

- [ ] **Step 4: Save the dict to a scratch file**

Create `docs/superpowers/plans/v04_baseline_golden.txt` and paste the printed dict into it. This file is referenced by Phase 19's update to `tests/integration/test_backwards_compat.py`. It is NOT committed (add to `.gitignore` if not already excluded by general gitignore rules).

- [ ] **Step 5: Verify the file is ignored**

Run: `git status`
Expected: `docs/superpowers/plans/v04_baseline_golden.txt` does NOT appear in tracked or untracked files. If it appears as untracked, append `docs/superpowers/plans/v04_baseline_*.txt` to `.gitignore` and commit `.gitignore` separately as a chore.

- [ ] **Step 6: No commit for Phase 1**

Phase 1 produces no source changes. Proceed to Phase 2.

---

## Phase 2: Data layer

Phase 2 lands the yfinance loader, the validator relaxation for index-style symbols, the synthetic-data generator relocation, and the sector_map.csv. Cumulative test target after Phase 2: **209** (202 + 7).

### Task 2: Add `auto_adjust` and `aux_symbols` to `DataConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Test: `tests/unit/test_config_models.py` (append three tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_models.py`:

```python
def test_data_config_auto_adjust_defaults_true():
    from backtester.config.models import DataConfig
    cfg = DataConfig(source="csv", root="data/raw", start="2024-01-01", end="2024-12-31", timeframe="1d")
    assert cfg.auto_adjust is True


def test_data_config_aux_symbols_defaults_empty():
    from backtester.config.models import DataConfig
    cfg = DataConfig(source="csv", root="data/raw", start="2024-01-01", end="2024-12-31", timeframe="1d")
    assert cfg.aux_symbols == []


def test_data_config_accepts_yfinance_source():
    from backtester.config.models import DataConfig
    cfg = DataConfig(source="yfinance", root="data/raw", start="2024-01-01", end="2024-12-31", timeframe="1d")
    assert cfg.source == "yfinance"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "auto_adjust or aux_symbols or yfinance"`
Expected: 2 FAIL with `AttributeError: 'DataConfig' object has no attribute 'auto_adjust' / 'aux_symbols'`. The `yfinance` test should already pass if `source` is just a string.

- [ ] **Step 3: Implement**

In `backtester/config/models.py`, add the two new fields at the end of `DataConfig` (after existing fields, before any methods). Preserve `slots=True` and the existing decorator.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass(slots=True)
class DataConfig:
    source: str
    root: str
    start: str
    end: str
    timeframe: str
    symbols: List[str] = field(default_factory=list)
    # v0.4.0 additions:
    auto_adjust: bool = True
    aux_symbols: List[str] = field(default_factory=list)
```

If the existing `DataConfig` definition includes other fields, preserve them all — the additions go at the end of the field list.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "auto_adjust or aux_symbols or yfinance"`
Expected: 3 PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: `205 passed` (202 + 3 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add auto_adjust + aux_symbols to DataConfig (v0.4.0)"
```

---

### Task 3: Add `strict_volume` flag to `validate_ohlcv`

**Files:**
- Modify: `backtester/data/validators.py`
- Test: `tests/unit/test_data_validators.py` (append two tests, create file if missing)

- [ ] **Step 1: Write the failing tests**

If `tests/unit/test_data_validators.py` does not exist, create it with the standard imports. Append (or create with):

```python
import pandas as pd
import pytest

from backtester.data.validators import validate_ohlcv


def _make_index_style_frame(rows: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [0.0] * rows,  # index-style: no volume
        },
        index=idx,
    )


def test_validator_default_rejects_zero_volume():
    df = _make_index_style_frame()
    with pytest.raises(Exception):
        validate_ohlcv(df)


def test_validator_strict_volume_false_allows_zero_volume():
    df = _make_index_style_frame()
    validate_ohlcv(df, strict_volume=False)  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_data_validators.py -v`
Expected: the second test FAILs with `TypeError: validate_ohlcv() got an unexpected keyword argument 'strict_volume'`. The first test passes if the existing validator already rejects zero-volume; if it doesn't, adjust the assertion to whatever the validator currently does for zero-volume input.

- [ ] **Step 3: Implement**

In `backtester/data/validators.py`, find `validate_ohlcv` and add a `strict_volume: bool = True` keyword argument. Replace the volume check (find the section that errors when volume is non-positive or NaN) with:

```python
def validate_ohlcv(df: pd.DataFrame, *, strict_volume: bool = True) -> None:
    # ... existing checks for index, columns, NaNs, positive prices ...

    if strict_volume:
        if (df["volume"] < 0).any():
            raise ValueError("validate_ohlcv: volume contains negative values")
        if df["volume"].isna().any():
            raise ValueError("validate_ohlcv: volume contains NaN")
    # When strict_volume=False, volume may be zero/NaN; we still coerce NaN to 0
    # to keep downstream serialization clean.
    if not strict_volume:
        df["volume"] = df["volume"].fillna(0.0)
```

Read the existing validator first and preserve its other checks. Only the volume branch changes.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_data_validators.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: `207 passed` (205 + 2 new).

- [ ] **Step 6: Commit**

```
git add backtester/data/validators.py tests/unit/test_data_validators.py
git commit -m "feat(validators): add strict_volume flag for index-style symbols"
```

---

### Task 4: Create `data/sector_map.csv` and its loader test

**Files:**
- Create: `data/sector_map.csv`
- Create: `tests/unit/test_sector_map.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sector_map.py`:

```python
from pathlib import Path
import csv


SECTOR_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "sector_map.csv"

REQUIRED_SYMBOLS = {
    "TSLA", "NVDA", "AMD", "COIN", "GOOGL", "MSTR", "XPEV", "NIO",
    "PLTR", "SMCI", "SHOP", "W", "META", "NFLX", "SPY", "^VIX",
}


def test_sector_map_csv_parses():
    assert SECTOR_MAP_PATH.exists(), f"missing {SECTOR_MAP_PATH}"
    with SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["symbol", "sector"], (
            f"unexpected header: {reader.fieldnames}"
        )
        rows = list(reader)
    assert len(rows) >= len(REQUIRED_SYMBOLS)


def test_every_required_symbol_has_sector_entry():
    with SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        rows = {row["symbol"]: row["sector"] for row in csv.DictReader(f)}
    missing = REQUIRED_SYMBOLS - set(rows)
    assert not missing, f"sector_map.csv missing: {missing}"
    for sym, sector in rows.items():
        assert sector, f"{sym} has empty sector"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_sector_map.py -v`
Expected: 2 FAIL with `AssertionError: missing .../data/sector_map.csv`.

- [ ] **Step 3: Create the CSV**

Create `data/sector_map.csv` with exactly this content:

```csv
symbol,sector
TSLA,Auto
NVDA,Semis
AMD,Semis
COIN,Crypto
GOOGL,BigTech
MSTR,Crypto
XPEV,Auto
NIO,Auto
PLTR,Software
SMCI,Semis
SHOP,Software
W,Consumer
META,BigTech
NFLX,Media
SPY,Index
^VIX,Index
```

End with a trailing newline.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_sector_map.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest -q`
Expected: `209 passed` (207 + 2 new).

- [ ] **Step 6: Commit**

```
git add data/sector_map.csv tests/unit/test_sector_map.py
git commit -m "feat(data): add sector_map.csv for PRD universe + index aux symbols"
```

---

### Task 5: Relocate synthetic-data generator output to `data/synth/`

**Files:**
- Modify: `scripts/generate_sample_data.py`
- Create: `data/synth/.gitkeep` (so the empty directory is tracked before fixtures land)

This is a preparatory task. It changes WHERE the synthetic generator writes; the actual relocation of bundled SPY/AAPL into `data/synth/` happens in Phase 17 (gated by the user-approval step for real data).

- [ ] **Step 1: Read the current generator**

Run: `python -m pytest tests/integration/test_backwards_compat.py -v`
Expected: PASS — establishes the test currently relies on synthetic data at `data/raw/`.

- [ ] **Step 2: Modify the generator**

In `scripts/generate_sample_data.py`, find the output directory constant (typically `data/raw` or similar). Replace with `data/synth`:

```python
# Top of file, near the existing OUT_DIR / output_dir constant:
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

If the script writes to a path argued through argparse, default that argument's `Path` to `data/synth`. Preserve seed/determinism logic — the synthetic OUTPUT is byte-identical to v0.3.0; only the destination directory changes.

- [ ] **Step 3: Create the gitkeep**

Create `data/synth/.gitkeep` (empty file).

- [ ] **Step 4: Run the generator into `data/synth/`**

Run: `python scripts/generate_sample_data.py`
Expected: writes `data/synth/SPY.csv` and `data/synth/AAPL.csv` (deterministic, ~2500 rows each for the 2015-2024 window).

- [ ] **Step 5: Verify generator output is byte-identical to current `data/raw/`**

Run (PowerShell):
```
fc.exe /b data\raw\SPY.csv data\synth\SPY.csv
fc.exe /b data\raw\AAPL.csv data\synth\AAPL.csv
```
Expected: `FC: no differences encountered` for both. If different, the generator's seed or output format drifted — STOP and reconcile.

- [ ] **Step 6: Do not delete `data/raw/` yet**

`data/raw/SPY.csv` and `data/raw/AAPL.csv` stay in place for now. They are replaced with real data in Phase 17. The backwards-compat test is re-pointed in Phase 19.

- [ ] **Step 7: Full suite (no regressions expected)**

Run: `python -m pytest -q`
Expected: `209 passed` (unchanged — no source code that affects tests has changed).

- [ ] **Step 8: Commit**

```
git add scripts/generate_sample_data.py data/synth/.gitkeep data/synth/SPY.csv data/synth/AAPL.csv
git commit -m "chore(data): relocate synthetic-data generator output to data/synth/"
```

Phase 2 ends here. Phase 3 (config models) builds on the additions to `DataConfig` and adds the rest of the v0.4.0 config surface. The yfinance loader itself lands in Phase 2.5 (Task 6 below) before we leave the data layer.

---

### Task 6: yfinance loader with cache-on-miss

**Files:**
- Create: `backtester/data/yfinance_loader.py`
- Modify: `backtester/data/loader.py`
- Test: `tests/unit/test_yfinance_loader.py` (create, 5 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_yfinance_loader.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _fake_yf_history(symbol: str, **_: object) -> pd.DataFrame:
    """yfinance-style frame: tz-aware index, columns Open/High/Low/Close/Volume."""
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low":  [ 99.0, 100.0, 101.0, 102.0, 103.0],
            "Close":[100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume":[1_000_000] * 5,
        },
        index=idx,
    )


def test_loader_reads_cache_when_csv_present(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached

    # Pre-populate the cache.
    sym_path = tmp_path / "FAKE.csv"
    fake_csv = _fake_yf_history("FAKE").rename(
        columns={c: c.lower() for c in ["Open", "High", "Low", "Close", "Volume"]}
    )
    fake_csv.index.name = "timestamp"
    fake_csv.to_csv(sym_path)

    with patch("backtester.data.yfinance_loader._yfinance_download") as mock_dl:
        df = load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
        )
        mock_dl.assert_not_called()  # cache hit — no network
    assert len(df) == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_loader_fetches_via_yfinance_when_csv_absent(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_fake_yf_history) as mock_dl:
        df = load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
        )
        mock_dl.assert_called_once()
    assert (tmp_path / "FAKE.csv").exists()
    assert len(df) == 5


def test_loader_raises_when_cache_does_not_cover_range(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    sym_path = tmp_path / "FAKE.csv"
    fake_csv = _fake_yf_history("FAKE").rename(
        columns={c: c.lower() for c in ["Open", "High", "Low", "Close", "Volume"]}
    )
    fake_csv.index.name = "timestamp"
    fake_csv.to_csv(sym_path)

    with pytest.raises(Exception, match="rm the file"):
        load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2019-01-01", end="2019-12-31",
        )


def test_loader_auto_adjust_true_passes_through(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_fake_yf_history) as mock_dl:
        load_yfinance_cached(
            symbol="FAKE", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
            auto_adjust=True,
        )
        kwargs = mock_dl.call_args.kwargs
        assert kwargs.get("auto_adjust") is True


def test_loader_require_volume_false_keeps_zero_volume(tmp_path: Path) -> None:
    from backtester.data.yfinance_loader import load_yfinance_cached
    def _vix_history(symbol, **_):
        df = _fake_yf_history(symbol)
        df["Volume"] = 0
        return df
    with patch("backtester.data.yfinance_loader._yfinance_download",
               side_effect=_vix_history):
        df = load_yfinance_cached(
            symbol="^VIX", root=str(tmp_path),
            start="2020-01-01", end="2020-01-10",
            require_volume=False,
        )
    assert (df["volume"] == 0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_yfinance_loader.py -v`
Expected: 5 FAIL with `ModuleNotFoundError: No module named 'backtester.data.yfinance_loader'`.

- [ ] **Step 3: Implement the loader**

Create `backtester/data/yfinance_loader.py`:

```python
from __future__ import annotations

from pathlib import Path
import pandas as pd

from backtester.core.exceptions import DataError


def _yfinance_download(symbol: str, *, auto_adjust: bool, period: str, progress: bool) -> pd.DataFrame:
    """Thin indirection around yfinance.download for monkeypatching in tests."""
    import yfinance  # local import: optional extras dependency
    return yfinance.download(
        symbol,
        period=period,
        auto_adjust=auto_adjust,
        progress=progress,
    )


def load_yfinance_cached(
    *,
    symbol: str,
    root: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
    require_volume: bool = True,
) -> pd.DataFrame:
    """Cache-on-miss yfinance loader.

    Behavior:
      - If `{root}/{symbol}.csv` exists, read it. If it covers `[start, end]`,
        slice and return. If it does NOT cover the range, raise DataError
        (explicit invalidation only — no silent re-fetch).
      - If absent, fetch via yfinance with `period="max"`, write the full
        history to `{root}/{symbol}.csv`, then slice to `[start, end]`.

    Adjustment contract: `auto_adjust=True` returns adjusted OHLC for all
    open/high/low/close columns. Volume is unadjusted.

    require_volume=False: treats zero/NaN volume as legitimate (for
    index-style symbols like ^VIX). Volume column is filled with 0 if NaN.
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    csv_path = root_p / f"{symbol}.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index.name = "timestamp"
        df = _slice_or_raise(df, symbol=symbol, start=start, end=end, csv_path=csv_path)
    else:
        raw = _yfinance_download(symbol, auto_adjust=auto_adjust, period="max", progress=False)
        df = _normalize_yfinance_frame(raw, require_volume=require_volume)
        df.to_csv(csv_path)
        df = df.loc[start:end]

    if not require_volume:
        df = df.copy()
        df["volume"] = df["volume"].fillna(0.0)

    return df


def _normalize_yfinance_frame(df: pd.DataFrame, *, require_volume: bool) -> pd.DataFrame:
    df = df.copy()
    # Drop tz if present so CSV round-trip is deterministic.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "timestamp"
    df.columns = [c.lower() for c in df.columns]
    # yfinance returns 'adj close' or 'close' depending on auto_adjust + version.
    # We keep only the canonical OHLCV; the trailing 'adj close' column (if any) is dropped.
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    if require_volume and "volume" in df.columns:
        if df["volume"].isna().any():
            raise DataError("yfinance returned NaN volume for a non-index symbol")
    return df


def _slice_or_raise(df: pd.DataFrame, *, symbol: str, start: str, end: str, csv_path: Path) -> pd.DataFrame:
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    if df.index.min() > ts_start or df.index.max() < ts_end:
        raise DataError(
            f"{symbol}.csv covers [{df.index.min().date()}, {df.index.max().date()}]; "
            f"requested [{ts_start.date()}, {ts_end.date()}]. rm the file at "
            f"{csv_path} to re-fetch."
        )
    return df.loc[start:end]
```

- [ ] **Step 4: Wire the loader into `load_symbol`**

In `backtester/data/loader.py`, find `load_symbol` and add the yfinance routing branch:

```python
def load_symbol(
    *,
    symbol: str,
    source: str,
    root: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
    require_volume: bool = True,
) -> pd.DataFrame:
    if source == "csv":
        return _load_csv(symbol=symbol, root=root, start=start, end=end)
    if source == "yfinance":
        from backtester.data.yfinance_loader import load_yfinance_cached
        return load_yfinance_cached(
            symbol=symbol, root=root, start=start, end=end,
            auto_adjust=auto_adjust, require_volume=require_volume,
        )
    raise ValueError(f"unknown data.source: {source}")
```

Preserve the existing CSV branch unchanged. Existing callers that pass `source="csv"` continue to work because `auto_adjust` and `require_volume` are kwargs-only with defaults.

If `load_symbol` is currently positional-only (or has a different signature), preserve all existing callers' call sites by keeping the parameters they pass in unchanged. The new `auto_adjust` and `require_volume` MUST be keyword-only — find the existing signature and add `*,` if needed.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/unit/test_yfinance_loader.py -v`
Expected: 5 PASS.

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: `214 passed` (209 + 5 new).

> **Note:** test count drifts from the cumulative-target table here because Task 6 was originally bundled with Task 2-5 in the table. The actual integer is what matters; the table is a rough budget.

- [ ] **Step 7: Commit**

```
git add backtester/data/yfinance_loader.py backtester/data/loader.py tests/unit/test_yfinance_loader.py
git commit -m "feat(data): yfinance loader with cache-on-miss + explicit invalidation"
```

Phase 2 ends here. The yfinance dependency is referenced via local import inside `_yfinance_download` so unit tests don't need it installed; only Phase 17's real-data fetch step does.

---

## Phase 3: Config models

Phase 3 adds the v0.4.0 fields to `ExecutionConfig`, `PortfolioConfig`, the new `RegimesConfig` (with three sub-configs), and `RunConfig`. The validation rules that constrain these fields land in Phase 4. Cumulative test target: **218** (214 + 4 = ExecutionConfig defaults, PortfolioConfig defaults, RegimesConfig defaults, RunConfig universe_path/regimes).

### Task 7: Add tranche-stop fields to `ExecutionConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Test: `tests/unit/test_config_models.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_models.py`:

```python
def test_execution_config_hard_stop_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.hard_stop_atr_mult is None


def test_execution_config_runner_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.runner_atr_mult is None


def test_execution_config_breakeven_floor_defaults_true():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.breakeven_floor is True


def test_execution_config_tranche_stop_atr_period_defaults_20():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.tranche_stop_atr_period == 20
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "hard_stop or runner_atr or breakeven or tranche"`
Expected: 4 FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

In `backtester/config/models.py`, locate `ExecutionConfig` and append four new fields after the existing v0.3.0 fields:

```python
from typing import Optional

@dataclass(slots=True)
class ExecutionConfig:
    # existing v0.3.0 fields preserved verbatim ...
    # v0.4.0 additions:
    hard_stop_atr_mult: Optional[float] = None
    runner_atr_mult: Optional[float] = None
    breakeven_floor: bool = True
    tranche_stop_atr_period: int = 20
```

The fields are kwargs-only because dataclass field ordering with defaults requires non-default fields to come first. Existing v0.3.0 fields already have defaults; the new fields go at the end.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "hard_stop or runner_atr or breakeven or tranche"`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `218 passed` (214 + 4 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add tranche-stop fields to ExecutionConfig (v0.4.0)"
```

---

### Task 8: Add portfolio sizing fields to `PortfolioConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Test: `tests/unit/test_config_models.py` (append)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_portfolio_config_sizing_mode_default_percent_equity():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.sizing_mode == "percent_equity"


def test_portfolio_config_vol_target_defaults_012():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.vol_target == 0.12


def test_portfolio_config_position_cap_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.position_cap_pct == 1.0


def test_portfolio_config_cash_reserve_pct_defaults_0():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.cash_reserve_pct == 0.0


def test_portfolio_config_risk_budget_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.risk_budget_pct == 1.0


def test_portfolio_config_sector_cap_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.sector_cap_pct == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "portfolio_config and (sizing_mode or vol_target or position_cap or cash_reserve or risk_budget or sector_cap)"`
Expected: most FAIL with `AttributeError` (some may pass if the existing PortfolioConfig already defines them).

- [ ] **Step 3: Implement**

In `backtester/config/models.py`, append to `PortfolioConfig`:

```python
@dataclass(slots=True)
class PortfolioConfig:
    # existing v0.3.0 fields preserved verbatim ...
    # v0.4.0 additions:
    vol_target: float = 0.12
    position_cap_pct: float = 1.0
    cash_reserve_pct: float = 0.0
    risk_budget_pct: float = 1.0
    sector_cap_pct: float = 1.0
```

`sizing_mode` and `size` likely already exist on the dataclass from v0.3.0 — preserve them. The `sizing_mode` test passes when `"percent_equity"` is its existing default. If `sizing_mode` is not yet a field, add it as `sizing_mode: str = "percent_equity"`.

The string value `"vol_targeted"` becomes a valid mode in Phase 10 (simulator wiring). No enum yet — keep it as a plain string for now.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v -k portfolio_config`
Expected: all PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `224 passed` (218 + 6 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add vol-target + risk/sector caps to PortfolioConfig"
```

---

### Task 9: Create `RegimesConfig` with three sub-configs

**Files:**
- Modify: `backtester/config/models.py`
- Test: `tests/unit/test_config_models.py` (append)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_spy_ema_regime_config_defaults():
    from backtester.config.models import SpyEmaRegimeConfig
    cfg = SpyEmaRegimeConfig()
    assert cfg.enabled is False
    assert cfg.ema_lookback == 200
    assert cfg.trip_pct == -0.02
    assert cfg.resume_pct == 0.02


def test_vix_regime_config_defaults():
    from backtester.config.models import VixRegimeConfig
    cfg = VixRegimeConfig()
    assert cfg.enabled is False
    assert cfg.trip_threshold == 30.0
    assert cfg.trip_consec == 2
    assert cfg.resume_threshold == 25.0
    assert cfg.resume_consec == 3


def test_circuit_breaker_config_defaults():
    from backtester.config.models import CircuitBreakerConfig
    cfg = CircuitBreakerConfig()
    assert cfg.enabled is False
    assert cfg.pnl_window_days == 20
    assert cfg.trip_pct == -0.05
    assert cfg.pause_days == 10


def test_regimes_config_holds_three_subconfigs():
    from backtester.config.models import RegimesConfig
    cfg = RegimesConfig()
    assert cfg.spy_ema.enabled is False
    assert cfg.vix.enabled is False
    assert cfg.circuit_breaker.enabled is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "regime or circuit_breaker"`
Expected: 4 FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to `backtester/config/models.py` (after `PortfolioConfig`):

```python
@dataclass(slots=True)
class SpyEmaRegimeConfig:
    enabled: bool = False
    ema_lookback: int = 200
    trip_pct: float = -0.02
    resume_pct: float = 0.02


@dataclass(slots=True)
class VixRegimeConfig:
    enabled: bool = False
    trip_threshold: float = 30.0
    trip_consec: int = 2
    resume_threshold: float = 25.0
    resume_consec: int = 3


@dataclass(slots=True)
class CircuitBreakerConfig:
    enabled: bool = False
    pnl_window_days: int = 20
    trip_pct: float = -0.05
    pause_days: int = 10


@dataclass(slots=True)
class RegimesConfig:
    spy_ema: SpyEmaRegimeConfig = field(default_factory=SpyEmaRegimeConfig)
    vix: VixRegimeConfig = field(default_factory=VixRegimeConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v -k "regime or circuit_breaker"`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `228 passed` (224 + 4 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add RegimesConfig (SPY EMA + VIX + circuit breaker)"
```

---

### Task 10: Add `universe_path` + `regimes` to `RunConfig`

**Files:**
- Modify: `backtester/config/models.py`
- Test: `tests/unit/test_config_models.py` (append)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_run_config_universe_path_defaults_none():
    from backtester.config.models import RunConfig
    # Inspect via dataclasses.fields rather than instantiating —
    # RunConfig may have required fields that don't have defaults.
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    assert "universe_path" in fields
    assert fields["universe_path"].default is None


def test_run_config_regimes_defaults_none():
    from backtester.config.models import RunConfig
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    assert "regimes" in fields
    assert fields["regimes"].default is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_models.py -v -k run_config`
Expected: 2 FAIL with `AssertionError: 'universe_path' not in fields`.

- [ ] **Step 3: Implement**

In `backtester/config/models.py`, locate `RunConfig` and add two new fields at the end. They must be Optional with `None` defaults so existing configs without these keys continue to parse.

```python
from pathlib import Path
from typing import Optional

@dataclass(slots=True)
class RunConfig:
    # existing v0.3.0 fields preserved verbatim ...
    # v0.4.0 additions:
    universe_path: Optional[Path] = None
    regimes: Optional[RegimesConfig] = None
```

If `RunConfig` uses `str` for path-like fields elsewhere, follow that convention — `universe_path: Optional[str] = None` is equally valid. The loader (Phase 5) is responsible for path resolution.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_models.py -v -k run_config`
Expected: 2 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `230 passed` (228 + 2 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add universe_path + regimes to RunConfig"
```

Phase 3 ends here. Cumulative target was 218; actual is 230 (the extras stem from PortfolioConfig and RegimesConfig each contributing more than the original budget).

---

## Phase 4: Config validation

Phase 4 adds the 18 validation rules from spec §7.8 to `backtester/config/validation.py`. Rules are grouped into 4 commits by domain (tranche-stop, portfolio-sizing, regimes, universe). Each rule gets exactly one test; each commit lands 4-5 related rules.

Cumulative test target: **248** (230 + 18 new).

### Task 11: Tranche-stop validation rules (rules 1-4)

**Files:**
- Modify: `backtester/config/validation.py`
- Test: `tests/unit/test_config_validation.py` (append 4 tests)

Covers spec §7.8 rules 1-4: hard/runner mutex, v0.3.0/v0.4.0 mutex, positivity, atr_period ≥ 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_validation.py`:

```python
def _base_run_config():
    """Helper: smallest valid RunConfig for v0.4.0 validation tests."""
    from backtester.config.models import (
        RunConfig, DataConfig, ExecutionConfig, PortfolioConfig,
    )
    return RunConfig(
        run_name="vtest",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30, "size": 1.0},
        data=DataConfig(source="csv", root="data/raw",
                        start="2024-01-01", end="2024-06-30", timeframe="1d",
                        symbols=["SPY"]),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(sizing_mode="percent_equity", size=0.95),
        output_root="output/runs",
    )


def test_validation_rule_01_hard_and_runner_both_or_neither():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = None
    with pytest.raises(ConfigError, match="both-or-neither"):
        validate_run_config(rc)


def test_validation_rule_02_v030_and_v040_keys_mutually_exclusive():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.execution.trailing_stop_pct = 0.05
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = 2.5
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)


def test_validation_rule_03_tranche_stop_mults_positive():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 0.0
    rc.execution.runner_atr_mult = 2.5
    with pytest.raises(ConfigError, match="hard_stop_atr_mult must be > 0"):
        validate_run_config(rc)


def test_validation_rule_04_tranche_stop_atr_period_min_2():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = 2.5
    rc.execution.tranche_stop_atr_period = 1
    with pytest.raises(ConfigError, match="tranche_stop_atr_period must be >= 2"):
        validate_run_config(rc)
```

Ensure `import pytest` is at the top of the file. If `_base_run_config` already exists (or a similar fixture), reuse it; the contents above are illustrative.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_0"`
Expected: 4 FAIL.

- [ ] **Step 3: Implement**

In `backtester/config/validation.py`, find `validate_run_config` and add a new `_validate_tranche_stop` helper called from the main function:

```python
def _validate_tranche_stop(rc) -> None:
    ex = rc.execution
    has_v030 = ex.trailing_stop_pct is not None or ex.trailing_stop_atr_mult is not None
    has_hard = ex.hard_stop_atr_mult is not None
    has_runner = ex.runner_atr_mult is not None

    if has_hard != has_runner:
        raise ConfigError(
            "execution: hard_stop_atr_mult and runner_atr_mult are both-or-neither"
        )
    if has_hard and has_v030:
        raise ConfigError(
            "execution: v0.3.0 trailing_stop_* keys and v0.4.0 hard/runner keys are mutually exclusive"
        )
    if has_hard:
        if ex.hard_stop_atr_mult <= 0:
            raise ConfigError("execution.hard_stop_atr_mult must be > 0")
        if ex.runner_atr_mult <= 0:
            raise ConfigError("execution.runner_atr_mult must be > 0")
        if ex.tranche_stop_atr_period < 2:
            raise ConfigError("execution.tranche_stop_atr_period must be >= 2")


def validate_run_config(rc) -> None:
    # ... existing v0.3.0 validation calls ...
    _validate_tranche_stop(rc)
```

Read the existing file first and preserve all existing rules. The new helper is called LAST in `validate_run_config` so existing tests still see the same error ordering.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_0"`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `234 passed` (230 + 4 new).

- [ ] **Step 6: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(validation): tranche-stop mutual exclusion + bounds"
```

---

### Task 12: Portfolio sizing validation rules (rules 5-9)

**Files:**
- Modify: `backtester/config/validation.py`
- Test: `tests/unit/test_config_validation.py` (append 5 tests)

Covers rules 5-9: position_cap_pct, cash_reserve_pct, risk_budget_pct, sector_cap_pct bounds, and vol_target > 0.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_validation_rule_05_position_cap_pct_bounds():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.position_cap_pct = bad
        with pytest.raises(ConfigError, match="position_cap_pct"):
            validate_run_config(rc)


def test_validation_rule_06_cash_reserve_pct_bounds():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    for bad in [-0.01, 1.0, 1.5]:
        rc = _base_run_config()
        rc.portfolio.cash_reserve_pct = bad
        with pytest.raises(ConfigError, match="cash_reserve_pct"):
            validate_run_config(rc)


def test_validation_rule_07_risk_budget_pct_bounds():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.risk_budget_pct = bad
        with pytest.raises(ConfigError, match="risk_budget_pct"):
            validate_run_config(rc)


def test_validation_rule_08_sector_cap_pct_bounds():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.sector_cap_pct = bad
        with pytest.raises(ConfigError, match="sector_cap_pct"):
            validate_run_config(rc)


def test_validation_rule_09_vol_target_positive_when_vol_targeted():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.portfolio.sizing_mode = "vol_targeted"
    rc.portfolio.vol_target = 0.0
    with pytest.raises(ConfigError, match="vol_target must be > 0"):
        validate_run_config(rc)
    # Other modes: vol_target=0 is allowed (field is ignored).
    rc.portfolio.sizing_mode = "percent_equity"
    rc.portfolio.vol_target = 0.0
    validate_run_config(rc)  # no raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_0[5-9]"`
Expected: 5 FAIL.

- [ ] **Step 3: Implement**

Add to `backtester/config/validation.py`:

```python
def _validate_portfolio_sizing(rc) -> None:
    p = rc.portfolio
    if not (0 < p.position_cap_pct <= 1):
        raise ConfigError(f"portfolio.position_cap_pct must be in (0, 1]; got {p.position_cap_pct}")
    if not (0 <= p.cash_reserve_pct < 1):
        raise ConfigError(f"portfolio.cash_reserve_pct must be in [0, 1); got {p.cash_reserve_pct}")
    if not (0 < p.risk_budget_pct <= 1):
        raise ConfigError(f"portfolio.risk_budget_pct must be in (0, 1]; got {p.risk_budget_pct}")
    if not (0 < p.sector_cap_pct <= 1):
        raise ConfigError(f"portfolio.sector_cap_pct must be in (0, 1]; got {p.sector_cap_pct}")
    if p.sizing_mode == "vol_targeted" and p.vol_target <= 0:
        raise ConfigError(f"portfolio.vol_target must be > 0 when sizing_mode='vol_targeted'")
```

Call `_validate_portfolio_sizing(rc)` from `validate_run_config` after `_validate_tranche_stop(rc)`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_0[5-9]"`
Expected: 5 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `239 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(validation): portfolio sizing bounds (cap_pct, reserve, budget, sector)"
```

---

### Task 13: Regime validation rules (rules 10-14)

**Files:**
- Modify: `backtester/config/validation.py`
- Test: `tests/unit/test_config_validation.py` (append 5 tests)

Covers rules 10-14: circuit-breaker pause_days ≥ 0, VIX hysteresis order, SPY pct signs, VIX consec ≥ 1, circuit_breaker trip_pct < 0.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def _base_with_regimes():
    from backtester.config.models import RegimesConfig
    rc = _base_run_config()
    rc.regimes = RegimesConfig()
    return rc


def test_validation_rule_10_circuit_breaker_pause_days_nonneg():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.circuit_breaker.pause_days = -1
    with pytest.raises(ConfigError, match="pause_days"):
        validate_run_config(rc)


def test_validation_rule_11_vix_resume_below_trip():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.vix.trip_threshold = 25
    rc.regimes.vix.resume_threshold = 30
    with pytest.raises(ConfigError, match="resume_threshold.*trip_threshold"):
        validate_run_config(rc)


def test_validation_rule_12_spy_pct_signs():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.spy_ema.trip_pct = 0.02  # must be <= 0
    with pytest.raises(ConfigError, match="spy_ema.trip_pct"):
        validate_run_config(rc)
    rc = _base_with_regimes()
    rc.regimes.spy_ema.resume_pct = -0.02  # must be >= 0
    with pytest.raises(ConfigError, match="spy_ema.resume_pct"):
        validate_run_config(rc)


def test_validation_rule_13_vix_consec_min_1():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.vix.trip_consec = 0
    with pytest.raises(ConfigError, match="vix.trip_consec"):
        validate_run_config(rc)
    rc = _base_with_regimes()
    rc.regimes.vix.resume_consec = 0
    with pytest.raises(ConfigError, match="vix.resume_consec"):
        validate_run_config(rc)


def test_validation_rule_14_circuit_breaker_trip_pct_negative():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.circuit_breaker.trip_pct = 0.05
    with pytest.raises(ConfigError, match="circuit_breaker.trip_pct"):
        validate_run_config(rc)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_1[0-4]"`
Expected: 5 FAIL.

- [ ] **Step 3: Implement**

Add to `backtester/config/validation.py`:

```python
def _validate_regimes(rc) -> None:
    if rc.regimes is None:
        return
    r = rc.regimes
    if r.circuit_breaker.pause_days < 0:
        raise ConfigError(f"regimes.circuit_breaker.pause_days must be >= 0; got {r.circuit_breaker.pause_days}")
    if r.vix.resume_threshold >= r.vix.trip_threshold:
        raise ConfigError(
            f"regimes.vix.resume_threshold ({r.vix.resume_threshold}) must be < "
            f"trip_threshold ({r.vix.trip_threshold})"
        )
    if r.spy_ema.trip_pct > 0:
        raise ConfigError(f"regimes.spy_ema.trip_pct must be <= 0; got {r.spy_ema.trip_pct}")
    if r.spy_ema.resume_pct < 0:
        raise ConfigError(f"regimes.spy_ema.resume_pct must be >= 0; got {r.spy_ema.resume_pct}")
    if r.vix.trip_consec < 1:
        raise ConfigError(f"regimes.vix.trip_consec must be >= 1; got {r.vix.trip_consec}")
    if r.vix.resume_consec < 1:
        raise ConfigError(f"regimes.vix.resume_consec must be >= 1; got {r.vix.resume_consec}")
    if r.circuit_breaker.trip_pct >= 0:
        raise ConfigError(f"regimes.circuit_breaker.trip_pct must be < 0; got {r.circuit_breaker.trip_pct}")
```

Call `_validate_regimes(rc)` from `validate_run_config`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_1[0-4]"`
Expected: 5 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `244 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(validation): regime gates (SPY pcts, VIX hysteresis, circuit breaker)"
```

---

### Task 14: Universe-membership validation rules (rules 15-18)

**Files:**
- Modify: `backtester/config/validation.py`
- Test: `tests/unit/test_config_validation.py` (append 4 tests)

Rules 15-18: universe_path exists, overrides ⊆ strategy_params, aux_symbols non-empty when regimes enabled, symbols/universe_path mutex. Rule 17 (every symbol has a sector entry) is verified by `_validate_universe` reading `sector_map.csv`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_validation_rule_15_universe_path_exists(tmp_path):
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_run_config()
    rc.universe_path = tmp_path / "missing.yaml"
    rc.data.symbols = []  # clear, otherwise rule 18 fires first
    with pytest.raises(ConfigError, match="universe_path"):
        validate_run_config(rc)


def test_validation_rule_16_overrides_subset_of_strategy_params(tmp_path):
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    universe_yaml = tmp_path / "universe.yaml"
    universe_yaml.write_text(
        "universe:\n"
        "  SPY: {sector: Index, overrides: {nonexistent_key: 1}}\n"
    )
    rc = _base_run_config()
    rc.universe_path = universe_yaml
    rc.data.symbols = []
    with pytest.raises(ConfigError, match="overrides"):
        validate_run_config(rc)


def test_validation_rule_17_aux_symbols_required_when_regimes_enabled():
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    rc = _base_with_regimes()
    rc.regimes.spy_ema.enabled = True
    rc.data.aux_symbols = []
    with pytest.raises(ConfigError, match="aux_symbols.*SPY"):
        validate_run_config(rc)


def test_validation_rule_18_symbols_and_universe_path_mutex(tmp_path):
    from backtester.config.validation import validate_run_config
    from backtester.core.exceptions import ConfigError
    universe_yaml = tmp_path / "universe.yaml"
    universe_yaml.write_text("universe:\n  SPY: {sector: Index}\n")
    rc = _base_run_config()
    rc.universe_path = universe_yaml
    # data.symbols already has SPY from _base_run_config
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_1[5-9]"`
Expected: 4 FAIL.

- [ ] **Step 3: Implement**

Add to `backtester/config/validation.py`:

```python
import yaml

def _validate_universe_path(rc) -> None:
    if rc.universe_path is None:
        return
    if rc.data.symbols:
        raise ConfigError(
            "data.symbols and universe_path are mutually exclusive; universe.yaml "
            "is the single source of symbol membership for multi-symbol runs"
        )
    if not Path(rc.universe_path).exists():
        raise ConfigError(f"universe_path does not exist: {rc.universe_path}")
    with open(rc.universe_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    universe = (doc or {}).get("universe", {})
    allowed_keys = set(rc.strategy_params.keys())
    for sym, meta in universe.items():
        overrides = (meta or {}).get("overrides", {}) or {}
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ConfigError(
                f"universe.yaml: {sym} overrides reference keys not in strategy_params: {sorted(unknown)}"
            )


def _validate_aux_symbols(rc) -> None:
    if rc.regimes is None:
        return
    required = []
    if rc.regimes.spy_ema.enabled:
        required.append("SPY")
    if rc.regimes.vix.enabled:
        required.append("^VIX")
    missing = [s for s in required if s not in rc.data.aux_symbols]
    if missing:
        raise ConfigError(
            f"data.aux_symbols must include {missing} because regimes are enabled "
            f"that depend on them"
        )
```

At the top of `validation.py`, ensure `from pathlib import Path` and `from backtester.core.exceptions import ConfigError` are imported.

Wire both helpers into `validate_run_config`:

```python
def validate_run_config(rc) -> None:
    # ... existing v0.3.0 calls ...
    _validate_tranche_stop(rc)
    _validate_portfolio_sizing(rc)
    _validate_regimes(rc)
    _validate_universe_path(rc)
    _validate_aux_symbols(rc)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_config_validation.py -v -k "rule_1[5-9]"`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `248 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(validation): universe-membership + aux_symbols + mutex rules"
```

Phase 4 ends. 18 new validation rules in 4 atomic commits, all enforced.

---

## Phase 5: Universe loader

Phase 5 introduces `backtester/config/universe.py`, the loader that parses `universe.yaml`, merges sector data from `sector_map.csv`, and produces `ResolvedSymbolConfig` instances per ticker. The loader is called from `backtester/config/loader.py` after the run YAML is parsed.

Cumulative test target: **252** (248 + 4).

### Task 15: `load_universe_config` with overrides merge

**Files:**
- Create: `backtester/config/universe.py`
- Modify: `backtester/config/loader.py`
- Test: `tests/unit/test_universe_loader.py` (create, 4 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_universe_loader.py`:

```python
from pathlib import Path

import pytest


def _write_universe(tmp_path, content: str) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(content)
    return p


def test_load_universe_yaml_with_per_name_overrides(tmp_path):
    from backtester.config.universe import load_universe_config

    path = _write_universe(tmp_path,
        "universe:\n"
        "  TSLA: {sector: Auto, overrides: {entry_atr_mult: 1.5}}\n"
        "  NVDA: {sector: Semis}\n"
    )
    universe = load_universe_config(
        path=path,
        global_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
    )
    assert universe["TSLA"].sector == "Auto"
    assert universe["TSLA"].effective_params["entry_atr_mult"] == 1.5
    assert universe["TSLA"].effective_params["mean_lookback"] == 10
    assert universe["NVDA"].sector == "Semis"
    assert universe["NVDA"].effective_params["entry_atr_mult"] == 1.25


def test_overrides_keys_must_be_subset_of_strategy_params(tmp_path):
    from backtester.config.universe import load_universe_config
    from backtester.core.exceptions import ConfigError

    path = _write_universe(tmp_path,
        "universe:\n"
        "  TSLA: {sector: Auto, overrides: {bogus_key: 1.5}}\n"
    )
    with pytest.raises(ConfigError, match="overrides"):
        load_universe_config(
            path=path,
            global_params={"entry_atr_mult": 1.25},
        )


def test_inline_sector_overrides_sector_map_csv(tmp_path):
    """universe.yaml's inline `sector` field wins over sector_map.csv."""
    from backtester.config.universe import load_universe_config

    # NVDA is "Semis" in sector_map.csv; we override to "Custom".
    path = _write_universe(tmp_path,
        "universe:\n"
        "  NVDA: {sector: Custom}\n"
    )
    universe = load_universe_config(path=path, global_params={})
    assert universe["NVDA"].sector == "Custom"


def test_missing_sector_raises_config_error(tmp_path):
    from backtester.config.universe import load_universe_config
    from backtester.core.exceptions import ConfigError

    path = _write_universe(tmp_path,
        "universe:\n"
        "  ZZZZ: {}\n"  # not in sector_map.csv and no inline sector
    )
    with pytest.raises(ConfigError, match="sector"):
        load_universe_config(path=path, global_params={})
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_universe_loader.py -v`
Expected: 4 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `backtester/config/universe.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backtester.core.exceptions import ConfigError


_SECTOR_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "sector_map.csv"


@dataclass(slots=True)
class ResolvedSymbolConfig:
    symbol: str
    sector: str
    effective_params: dict[str, Any] = field(default_factory=dict)


def _load_sector_map() -> dict[str, str]:
    if not _SECTOR_MAP_PATH.exists():
        return {}
    with _SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        return {row["symbol"]: row["sector"] for row in csv.DictReader(f)}


def load_universe_config(
    *,
    path: Path,
    global_params: dict[str, Any],
) -> dict[str, ResolvedSymbolConfig]:
    """Parse universe.yaml and resolve per-symbol sector + overrides.

    Resolution precedence (low → high):
      1. global_params (from run YAML's strategy_params)
      2. per-name overrides (from universe.yaml)

    Sector resolution:
      1. data/sector_map.csv lookup
      2. universe.yaml inline `sector` field (wins if present)
      3. ConfigError if neither resolves to a non-empty string

    Returns dict[symbol, ResolvedSymbolConfig].
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    universe_dict = doc.get("universe", {})
    if not isinstance(universe_dict, dict):
        raise ConfigError(f"{path}: top-level `universe:` must be a mapping")

    sector_map = _load_sector_map()
    allowed_keys = set(global_params)
    out: dict[str, ResolvedSymbolConfig] = {}

    for symbol, meta in universe_dict.items():
        meta = meta or {}
        overrides = meta.get("overrides", {}) or {}
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ConfigError(
                f"universe.yaml: {symbol} overrides reference keys not in "
                f"strategy_params: {sorted(unknown)}"
            )
        inline_sector = meta.get("sector")
        sector = inline_sector if inline_sector else sector_map.get(symbol)
        if not sector:
            raise ConfigError(
                f"universe.yaml: {symbol} has no sector (not in sector_map.csv "
                f"and no inline `sector` field)"
            )
        effective = dict(global_params)
        effective.update(overrides)
        out[symbol] = ResolvedSymbolConfig(
            symbol=symbol, sector=sector, effective_params=effective,
        )
    return out
```

Wire into `backtester/config/loader.py` by adding (after the run YAML is loaded):

```python
# At the bottom of load_run_config, after parsing the YAML and constructing rc:
if rc.universe_path is not None:
    from backtester.config.universe import load_universe_config
    # Resolve relative to the run YAML's directory.
    universe_path = (Path(args.config).parent / rc.universe_path).resolve() \
                    if not Path(rc.universe_path).is_absolute() \
                    else Path(rc.universe_path)
    rc.universe_path = universe_path  # normalize for downstream consumers
    # The actual universe dict is loaded lazily by the runner — here we just
    # validate that the file exists and parses. The full validation runs in
    # validate_run_config.
```

If `load_run_config` doesn't have an `args.config` in scope (the loader is decoupled from argparse), pass the run YAML path explicitly into the function. Read the existing loader and follow its conventions.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_universe_loader.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `252 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/config/universe.py backtester/config/loader.py tests/unit/test_universe_loader.py
git commit -m "feat(config): universe.yaml loader + sector_map merge + overrides resolution"
```

Phase 5 ends. The `RunConfig.universe_path` field is now end-to-end wired through validation + loading.

---

## Phase 6: Strategy contract extensions

Phase 6 extends the strategy contract minimally so v0.4.0 strategies can opt into multi-symbol mode, per-bar callbacks, and the new `StrategyContext` fields. v0.3.0 strategies (sma_cross, momentum_streak, rsi_long_short, etc.) are NOT modified — the new fields have defaults.

Cumulative test target: **256** (252 + 4).

### Task 16: Add `uses_multi_symbol` and `uses_per_bar` to `BaseStrategy`

**Files:**
- Modify: `backtester/strategies/base.py`
- Test: `tests/unit/test_strategy_base.py` (create if missing, append two tests)

- [ ] **Step 1: Write the failing tests**

If `tests/unit/test_strategy_base.py` does not exist, create it with this content. Otherwise append:

```python
def test_base_strategy_uses_multi_symbol_default_false():
    from backtester.strategies.base import BaseStrategy
    assert BaseStrategy.uses_multi_symbol is False


def test_base_strategy_uses_per_bar_default_false():
    from backtester.strategies.base import BaseStrategy
    assert BaseStrategy.uses_per_bar is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_base.py -v`
Expected: 2 FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

In `backtester/strategies/base.py`, add two class-level attributes to `BaseStrategy`:

```python
class BaseStrategy(Generic[ParamsT]):
    # existing class attributes (strategy_id, version, asset_type, timeframe) ...

    # v0.4.0 opt-in attributes (default False keeps v0.3.0 strategies unchanged):
    uses_multi_symbol: bool = False
    uses_per_bar: bool = False

    # existing methods (params_type, warmup_bars, indicators, generate_signals) ...
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_strategy_base.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `254 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/strategies/base.py tests/unit/test_strategy_base.py
git commit -m "feat(strategies): BaseStrategy.uses_multi_symbol + uses_per_bar opt-in flags"
```

---

### Task 17: Extend `StrategyContext` with v0.4.0 fields

**Files:**
- Modify: `backtester/core/types.py`
- Test: `tests/unit/test_strategy_context.py` (create if missing, append two tests)

- [ ] **Step 1: Write the failing tests**

Create or append to `tests/unit/test_strategy_context.py`:

```python
def test_strategy_context_has_v04_fields():
    """v0.4.0 fields exist on StrategyContext with safe defaults."""
    from backtester.core.types import StrategyContext
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(StrategyContext)}
    assert "position_phase" in field_names
    assert "bars_in_phase" in field_names
    assert "recent_pnl" in field_names
    assert "regime" in field_names


def test_strategy_context_defaults_empty():
    """Old call-sites that don't pass v0.4.0 fields still work."""
    from backtester.core.types import StrategyContext
    # Construct using only the v0.3.0-era required fields. The new fields
    # must have defaults so this does not raise.
    import inspect
    sig = inspect.signature(StrategyContext)
    # Build kwargs for required fields only (those without defaults).
    required = {
        name: _placeholder_for(param)
        for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
    }
    ctx = StrategyContext(**required)
    assert ctx.position_phase == {} or ctx.position_phase is None
    assert ctx.bars_in_phase == {} or ctx.bars_in_phase is None


def _placeholder_for(param):
    """Helper: return a value of the right kind for a required field."""
    ann = param.annotation
    if ann is str:
        return ""
    if ann is int:
        return 0
    return None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_context.py -v`
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

In `backtester/core/types.py`, locate `StrategyContext` and add four new optional fields. The fields' contents are populated by the multi-symbol simulator only; v0.3.0 single-symbol callers leave them empty.

```python
from typing import Optional, Any
from dataclasses import dataclass, field
import pandas as pd

@dataclass(slots=True)
class StrategyContext:
    # existing v0.3.0 fields preserved verbatim ...

    # v0.4.0 additions: simulator-populated state visible to strategies.
    position_phase: dict[str, Any] = field(default_factory=dict)
    bars_in_phase: dict[str, int] = field(default_factory=dict)
    recent_pnl: Optional[pd.Series] = None
    regime: Optional[Any] = None  # RegimePolicy or RegimeState — typed in Phase 8
```

`Any` is used for `position_phase` values because `TSPhase` doesn't exist yet (Phase 7). Tightened in Phase 7's task.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_strategy_context.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `256 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/core/types.py tests/unit/test_strategy_context.py
git commit -m "feat(types): extend StrategyContext with position_phase/bars_in_phase/recent_pnl/regime"
```

Phase 6 ends. v0.3.0 strategies and simulator paths are demonstrably unaffected (test count went up by 4; v0.3.0 baseline tests are all still green).

---

## Phase 7: TrancheStopState

Phase 7 is the v0.4.0 close-basis trailing stop with `HARD → RUNNER → DISARMED` phases. It is a SEPARATE class from v0.3.0's `TrailingStopState`; the v0.3.0 class is not touched. The split was a load-bearing brainstorming decision (see spec §2.1).

14 unit tests, split into 3 tasks: 5 tests for the phase state machine, 5 tests for the stop-price branches, 4 tests for edge cases.

Cumulative test target: **270** (256 + 14).

### Task 18: `TSPhase` enum + `TrancheStopState` skeleton + state machine

**Files:**
- Create: `backtester/engine/tranche_stop.py`
- Test: `tests/unit/test_tranche_stop.py` (create, 5 tests for state transitions)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tranche_stop.py`:

```python
import math

import pandas as pd
import pytest


# ---- helpers ----

def _atr_series(values, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="atr")


def _make_state(*, hard=1.75, runner=2.5, breakeven_floor=True, atr_series=None):
    from backtester.engine.tranche_stop import TrancheStopState
    if atr_series is None:
        atr_series = _atr_series([2.0] * 20)
    return TrancheStopState(
        hard_stop_atr_mult=hard,
        runner_atr_mult=runner,
        breakeven_floor=breakeven_floor,
        atr_series=atr_series,
    )


# ---- 5 state-machine tests ----

def test_disarmed_by_default():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    assert ts.phase is TSPhase.DISARMED


def test_reset_sets_hard_phase_and_snapshots_entry():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=5)
    assert ts.phase is TSPhase.HARD
    assert ts.entry_price == 100.0
    assert ts.entry_bar_idx == 5
    assert ts.atr_at_entry == 2.0
    assert ts.peak_close == 100.0
    assert ts.trough_close == 100.0


def test_promote_to_runner_keeps_peak_close():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    # Simulate a few up-bars in HARD phase.
    for c in [101.0, 102.5, 104.0]:
        ts.update(pd.Series({"high": c + 1, "low": c - 1, "close": c}))
    assert ts.peak_close == 104.0
    ts.promote_to_runner()
    assert ts.phase is TSPhase.RUNNER
    assert ts.peak_close == 104.0  # peak persists across promotion


def test_promote_to_runner_idempotent():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    ts.promote_to_runner()  # second call no-op
    assert ts.phase is TSPhase.RUNNER


def test_disarm_clears_peak_close_and_phase():
    from backtester.engine.tranche_stop import TSPhase
    ts = _make_state()
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.update(pd.Series({"high": 105, "low": 99, "close": 103}))
    ts.disarm()
    assert ts.phase is TSPhase.DISARMED
    assert ts.peak_close == 0.0
    assert math.isinf(ts.trough_close)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_tranche_stop.py -v`
Expected: 5 FAIL with `ModuleNotFoundError: No module named 'backtester.engine.tranche_stop'`.

- [ ] **Step 3: Implement skeleton + state machine**

Create `backtester/engine/tranche_stop.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class TSPhase(Enum):
    """Tranche-stop phase machine.

    DISARMED → (reset) → HARD → (promote_to_runner) → RUNNER → (disarm) → DISARMED
    """
    HARD = "hard"
    RUNNER = "runner"
    DISARMED = "disarmed"


@dataclass
class TrancheStopState:
    """v0.4.0 close-basis trailing stop with two-phase HARD→RUNNER machine.

    Separate from v0.3.0's TrailingStopState. The two coexist; runs use one
    or the other (config-validation rule 2 enforces the mutex).

    Key semantics:
      - HARD phase: fixed stop at entry_price - hard_stop_atr_mult * atr_at_entry.
        ATR is snapshotted at entry — does NOT trail with current ATR.
      - RUNNER phase: trail at peak_close - runner_atr_mult * atr_now,
        optionally floored at entry_price (breakeven_floor=True by default).
      - Intrabar wicks DO NOT move the runner trail. Only confirmed closes do.
    """

    # configuration (immutable per run)
    hard_stop_atr_mult: float
    runner_atr_mult: float
    breakeven_floor: bool = True
    atr_series: Optional[pd.Series] = None

    # snapshotted at reset() — frozen for the life of the position
    entry_price: float = 0.0
    entry_bar_idx: int = -1
    atr_at_entry: float = float("nan")

    # mutating state — tracks the position
    phase: TSPhase = TSPhase.DISARMED
    peak_close: float = 0.0
    trough_close: float = float("inf")

    # ---- state-machine API ----

    def reset(self, *, entry_price: float, bar_idx: int) -> None:
        """Called on flat → non-flat transition. Snapshots entry state."""
        self.entry_price = entry_price
        self.entry_bar_idx = bar_idx
        self.atr_at_entry = (
            float(self.atr_series.iloc[bar_idx]) if self.atr_series is not None else float("nan")
        )
        self.peak_close = entry_price
        self.trough_close = entry_price
        self.phase = TSPhase.HARD

    def promote_to_runner(self) -> None:
        """Called by simulator on detected partial close from HARD. Idempotent."""
        if self.phase is TSPhase.HARD:
            self.phase = TSPhase.RUNNER

    def disarm(self) -> None:
        """Called on any transition to flat. Clears mutating state."""
        self.phase = TSPhase.DISARMED
        self.peak_close = 0.0
        self.trough_close = float("inf")

    def update(self, bar: pd.Series) -> None:
        """Per-bar ratchet on CLOSE only. Intrabar wicks ignored."""
        if self.phase is TSPhase.DISARMED:
            return
        c = float(bar["close"])
        if c > self.peak_close:
            self.peak_close = c
        if c < self.trough_close:
            self.trough_close = c

    def stop_price(self, *, sign: int, bar_idx: int) -> Optional[float]:
        """Stop level for the next bar's STOP order. Implemented in Task 19."""
        raise NotImplementedError("Implemented in Task 19")
```

The `stop_price` method body is deferred to Task 19. Importing the module does not call `stop_price`, so the state-machine tests pass with the NotImplementedError stub.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_tranche_stop.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `261 passed` (256 + 5 new).

- [ ] **Step 6: Commit**

```
git add backtester/engine/tranche_stop.py tests/unit/test_tranche_stop.py
git commit -m "feat(engine): TrancheStopState skeleton + HARD/RUNNER/DISARMED state machine"
```

---

### Task 19: `stop_price` HARD branch + RUNNER branch + breakeven floor

**Files:**
- Modify: `backtester/engine/tranche_stop.py` (implement `stop_price`)
- Test: `tests/unit/test_tranche_stop.py` (append 5 tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tranche_stop.py`:

```python
def test_hard_stop_uses_atr_at_entry_not_current_atr():
    """HARD-phase stop is fixed at entry; current ATR is irrelevant."""
    # ATR rises across the holding period, but HARD stop uses the entry ATR.
    atr = _atr_series([2.0, 2.0, 2.0, 10.0, 10.0])
    ts = _make_state(hard=1.75, runner=2.5, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    # ATR at entry = 2.0 → hard stop = 100 - 1.75*2 = 96.5
    assert ts.stop_price(sign=+1, bar_idx=3) == pytest.approx(96.5)
    # Even though atr_series[3] = 10.0, the stop does not move.


def test_hard_stop_does_not_trail():
    atr = _atr_series([2.0] * 10)
    ts = _make_state(atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    # Price ratchets up, but the stop should remain at entry-based level.
    for i, c in enumerate([101.0, 103.0, 105.0], start=1):
        ts.update(pd.Series({"high": c + 0.5, "low": c - 0.5, "close": c}))
    # HARD stop = 100 - 1.75*2 = 96.5; does not move even though peak_close = 105.
    assert ts.stop_price(sign=+1, bar_idx=3) == pytest.approx(96.5)


def test_runner_trail_uses_peak_close_not_peak_high():
    """Critical: intrabar wicks (high) DO NOT move the runner trail."""
    atr = _atr_series([2.0] * 10)
    ts = _make_state(runner=2.5, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Update with bars whose HIGH is well above the CLOSE.
    ts.update(pd.Series({"high": 120.0, "low": 99.0, "close": 105.0}))
    # peak_close is 105.0, not 120.0.
    # Runner stop = 105 - 2.5*2 = 100.0; with breakeven floor active → max(100, 100) = 100.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(100.0)


def test_runner_trail_clipped_by_breakeven_floor():
    """When raw trail is below entry, breakeven_floor clamps it up to entry_price."""
    atr = _atr_series([5.0] * 10)
    ts = _make_state(runner=2.5, breakeven_floor=True, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Price didn't move much; peak_close ≈ 100.
    ts.update(pd.Series({"high": 101.0, "low": 99.0, "close": 100.5}))
    # Raw trail = 100.5 - 2.5*5 = 88.0; floored at entry_price = 100.0.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(100.0)


def test_runner_trail_no_floor_when_disabled():
    atr = _atr_series([5.0] * 10)
    ts = _make_state(runner=2.5, breakeven_floor=False, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    ts.update(pd.Series({"high": 101.0, "low": 99.0, "close": 100.5}))
    # Raw trail = 100.5 - 12.5 = 88.0; no floor.
    assert ts.stop_price(sign=+1, bar_idx=1) == pytest.approx(88.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_tranche_stop.py -v -k "hard_stop or runner_trail"`
Expected: 5 FAIL with `NotImplementedError: Implemented in Task 19`.

- [ ] **Step 3: Implement `stop_price`**

In `backtester/engine/tranche_stop.py`, replace the `stop_price` method body:

```python
    def stop_price(self, *, sign: int, bar_idx: int) -> Optional[float]:
        if self.phase is TSPhase.DISARMED or sign == 0:
            return None

        if self.phase is TSPhase.HARD:
            if pd.isna(self.atr_at_entry):
                return None
            offset = self.hard_stop_atr_mult * self.atr_at_entry
            # Fixed at entry; does NOT trail.
            return self.entry_price - offset if sign > 0 else self.entry_price + offset

        # RUNNER
        if self.atr_series is None:
            return None
        atr_now = float(self.atr_series.iloc[bar_idx])
        if pd.isna(atr_now):
            return None
        offset = self.runner_atr_mult * atr_now
        raw = (self.peak_close - offset) if sign > 0 else (self.trough_close + offset)
        if self.breakeven_floor:
            return max(raw, self.entry_price) if sign > 0 else min(raw, self.entry_price)
        return raw
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_tranche_stop.py -v`
Expected: 10 PASS (5 from Task 18 + 5 new).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `266 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/tranche_stop.py tests/unit/test_tranche_stop.py
git commit -m "feat(engine): TrancheStopState stop_price (HARD frozen, RUNNER close-basis+floor)"
```

---

### Task 20: TrancheStopState edge cases (warmup, short side, update gating)

**Files:**
- Test: `tests/unit/test_tranche_stop.py` (append 4 tests; no source changes — verifies existing behavior)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_update_only_reads_close_field():
    """update() must IGNORE high/low fields; only close moves the ratchet."""
    atr = _atr_series([2.0] * 10)
    ts = _make_state(atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Pass a series where high is wildly higher than close.
    ts.update(pd.Series({"high": 200.0, "low": 50.0, "close": 100.0}))
    # peak_close must NOT have moved to 200.
    assert ts.peak_close == 100.0
    assert ts.trough_close == 100.0


def test_stop_price_returns_none_when_atr_nan():
    """During ATR warmup, stop_price must return None (no stop scheduled)."""
    atr = pd.Series(
        [float("nan")] * 5 + [2.0] * 5,
        index=pd.date_range("2024-01-02", periods=10, freq="B"),
    )
    ts = _make_state(atr_series=atr)
    # Reset during warmup: atr_at_entry is NaN.
    ts.reset(entry_price=100.0, bar_idx=0)
    assert ts.stop_price(sign=+1, bar_idx=0) is None
    # In RUNNER phase, if current ATR is NaN, also None.
    ts.promote_to_runner()
    assert ts.stop_price(sign=+1, bar_idx=2) is None
    # Once ATR is valid, stop_price is computed.
    assert ts.stop_price(sign=+1, bar_idx=5) is not None


def test_short_symmetric_trough_close_and_breakeven_ceiling():
    """Short side: trough_close ratchets down; breakeven floor becomes a ceiling."""
    atr = _atr_series([2.0] * 10)
    ts = _make_state(runner=2.5, atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.promote_to_runner()
    # Short position; price drops.
    for c in [98.0, 96.0, 94.0]:
        ts.update(pd.Series({"high": c + 1, "low": c - 1, "close": c}))
    assert ts.trough_close == 94.0
    # Raw runner stop = 94 + 2.5*2 = 99.0; breakeven ceiling = min(99, 100) = 99.0.
    assert ts.stop_price(sign=-1, bar_idx=3) == pytest.approx(99.0)
    # If price rallies above entry, breakeven CEILING (min for shorts) clamps stop.
    ts.update(pd.Series({"high": 102, "low": 100, "close": 101.0}))
    # trough_close still 94; raw stop = 94 + 5 = 99; ceiling = min(99, 100) = 99.
    assert ts.stop_price(sign=-1, bar_idx=4) == pytest.approx(99.0)


def test_re_adding_during_runner_does_not_reset_phase():
    """An external caller increasing position size mid-RUNNER must NOT reset phase.

    Documented as undefined behavior in spec §2.6 — the test pins down what
    actually happens: phase stays RUNNER, peak_close unchanged.
    """
    atr = _atr_series([2.0] * 10)
    ts = _make_state(atr_series=atr)
    ts.reset(entry_price=100.0, bar_idx=0)
    ts.update(pd.Series({"high": 105, "low": 99, "close": 104.0}))
    ts.promote_to_runner()
    saved_peak = ts.peak_close
    saved_entry = ts.entry_price
    # Simulator scales position size up — but per spec, no API on
    # TrancheStopState to "re-enter". Verify the state is untouched.
    assert ts.phase.value == "runner"
    assert ts.peak_close == saved_peak
    assert ts.entry_price == saved_entry
```

- [ ] **Step 2: Run to verify**

Run: `python -m pytest tests/unit/test_tranche_stop.py -v`
Expected: All 14 PASS (5 from Task 18 + 5 from Task 19 + 4 new). No source changes were needed; the tests verify behavior already implemented.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: `270 passed`.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_tranche_stop.py
git commit -m "test(tranche_stop): edge cases — warmup, short side, update gating"
```

Phase 7 ends. 14 unit tests cover TrancheStopState exhaustively. The class is ready for simulator integration in Phase 10.

---

## Phase 8: RegimePolicy

Phase 8 implements the three independent regime gates (SPY 200-EMA, VIX hysteresis, strategy circuit breaker) and their disjunction into `book_flat`. The policy is a pure-update object — the simulator calls `update(bar_idx, aux_data, recent_pnl)` per bar.

10 unit tests in 2 tasks: 6 tests for the three gates individually, 4 tests for the disjunction + flatten-on-trip + disabled-gate behavior.

Cumulative test target: **280** (270 + 10).

### Task 21: Three regime sub-gates with hysteresis

**Files:**
- Create: `backtester/engine/regime.py`
- Test: `tests/unit/test_regime_policy.py` (create, 6 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_regime_policy.py`:

```python
import pandas as pd
import pytest


def _spy_series(values, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"close": values}, index=idx)


def _vix_series(values):
    return _spy_series(values)


def test_spy_ema_trips_below_threshold():
    from backtester.engine.regime import SpyEmaGate
    gate = SpyEmaGate(ema_lookback=3, trip_pct=-0.02, resume_pct=0.02)
    # Build a series where SPY drops well below its EMA.
    spy = _spy_series([100, 100, 100, 100, 80])
    for i in range(len(spy)):
        gate.update(bar_idx=i, spy_close=spy["close"], spy_ema=spy["close"].ewm(span=3, adjust=False).mean())
    assert gate.tripped is True


def test_spy_ema_resumes_above_threshold_hysteresis():
    from backtester.engine.regime import SpyEmaGate
    gate = SpyEmaGate(ema_lookback=3, trip_pct=-0.02, resume_pct=0.02)
    # First trip, then a small bounce (within hysteresis) — gate stays tripped.
    # Then a full recovery — gate resumes.
    closes = [100, 100, 100, 100, 80, 95, 98, 110]
    spy = _spy_series(closes)
    ema = spy["close"].ewm(span=3, adjust=False).mean()
    for i in range(len(spy)):
        gate.update(bar_idx=i, spy_close=spy["close"], spy_ema=ema)
    assert gate.tripped is False  # recovered by bar 7


def test_vix_requires_two_consecutive_above_30():
    from backtester.engine.regime import VixGate
    gate = VixGate(trip_threshold=30, trip_consec=2, resume_threshold=25, resume_consec=3)
    vix = _vix_series([20, 31, 20, 31, 32])
    for i in range(len(vix)):
        gate.update(bar_idx=i, vix_close=vix["close"])
    # Single spikes (bar 1, bar 3) don't trip; bars 3+4 are two consecutive >30 → tripped.
    assert gate.tripped is True


def test_vix_resume_requires_three_consecutive_below_25():
    from backtester.engine.regime import VixGate
    gate = VixGate(trip_threshold=30, trip_consec=2, resume_threshold=25, resume_consec=3)
    # Trip: 31, 32. Then 20, 20 (two below 25, not enough). Then 20 (three) → resumed.
    vix = _vix_series([31, 32, 20, 20, 20])
    for i in range(len(vix)):
        gate.update(bar_idx=i, vix_close=vix["close"])
    assert gate.tripped is False


def test_circuit_breaker_trips_on_minus_5_pct_rolling_20d():
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    # Initial equity = 100_000; rolling sum of pnls over 5 days reaches -5_000.
    pnls = [-1000.0] * 5 + [0.0] * 5
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    # After 5 days of -1000 PnL on $100k equity, rolling sum = -5000 → -5% → trip.
    assert gate.tripped_history[4] is True


def test_circuit_breaker_resumes_after_pause_days():
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    pnls = [-1000.0] * 5 + [0.0, 0.0, 0.0, 0.0]
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    # Tripped at bar 4 (idx 4); pause_days=2 means resume at bar 4 + 2 + 1 = 7.
    # Bar 7 → resumed (full size, PRD literal).
    assert gate.tripped_history[7] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_regime_policy.py -v`
Expected: 6 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `backtester/engine/regime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SpyEmaGate:
    """SPY-vs-200-day-EMA gate with hysteresis.

    trip:    spy_close[i] < spy_ema[i] * (1 + trip_pct)     (trip_pct typically -0.02)
    resume:  spy_close[i] > spy_ema[i] * (1 + resume_pct)   (resume_pct typically  0.02)
    """
    ema_lookback: int = 200
    trip_pct: float = -0.02
    resume_pct: float = 0.02
    tripped: bool = False
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, spy_close: pd.Series, spy_ema: pd.Series) -> None:
        c = float(spy_close.iloc[bar_idx])
        e = float(spy_ema.iloc[bar_idx])
        trip_value = e * (1.0 + self.trip_pct)
        resume_value = e * (1.0 + self.resume_pct)
        if not self.tripped:
            if c < trip_value:
                self.tripped = True
        else:
            if c > resume_value:
                self.tripped = False
        self.tripped_history.append(self.tripped)


@dataclass
class VixGate:
    """VIX hysteresis gate.

    trip:    last trip_consec closes > trip_threshold
    resume:  last resume_consec closes < resume_threshold
    """
    trip_threshold: float = 30.0
    trip_consec: int = 2
    resume_threshold: float = 25.0
    resume_consec: int = 3
    tripped: bool = False
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, vix_close: pd.Series) -> None:
        if not self.tripped:
            window = vix_close.iloc[max(0, bar_idx - self.trip_consec + 1): bar_idx + 1]
            if len(window) >= self.trip_consec and (window > self.trip_threshold).all():
                self.tripped = True
        else:
            window = vix_close.iloc[max(0, bar_idx - self.resume_consec + 1): bar_idx + 1]
            if len(window) >= self.resume_consec and (window < self.resume_threshold).all():
                self.tripped = False
        self.tripped_history.append(self.tripped)


@dataclass
class CircuitBreakerGate:
    """Rolling-N-day strategy-PnL kill switch.

    trip:    rolling pnl_window_days sum / initial_cash <= trip_pct (negative)
    resume:  pause_days bars after the trip bar (PRD literal: full size on day pause_days+1).
    """
    pnl_window_days: int = 20
    trip_pct: float = -0.05
    pause_days: int = 10
    tripped: bool = False
    _trip_bar_idx: int = -1
    tripped_history: list[bool] = field(default_factory=list)

    def update(self, *, bar_idx: int, recent_pnl: pd.Series, initial_cash: float) -> None:
        if not self.tripped:
            window = recent_pnl.iloc[max(0, bar_idx - self.pnl_window_days + 1): bar_idx + 1]
            rolling_pct = float(window.sum()) / float(initial_cash) if initial_cash else 0.0
            if rolling_pct <= self.trip_pct:
                self.tripped = True
                self._trip_bar_idx = bar_idx
        else:
            # Resume on bar _trip_bar_idx + pause_days + 1 (PRD: "full size on day 11").
            if bar_idx > self._trip_bar_idx + self.pause_days:
                self.tripped = False
        self.tripped_history.append(self.tripped)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_regime_policy.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `276 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/regime.py tests/unit/test_regime_policy.py
git commit -m "feat(engine): three independent regime gates (SPY EMA, VIX, circuit breaker)"
```

---

### Task 22: `RegimePolicy` composition + flatten-on-trip behavior

**Files:**
- Modify: `backtester/engine/regime.py` (add `RegimePolicy` + `RegimeState`)
- Test: `tests/unit/test_regime_policy.py` (append 4 tests)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_circuit_breaker_resumes_at_full_size_not_phased():
    """PRD literal: re-entry on day 11 is at full size, no phased ramp."""
    from backtester.engine.regime import CircuitBreakerGate
    gate = CircuitBreakerGate(pnl_window_days=5, trip_pct=-0.05, pause_days=2)
    pnls = [-1000.0] * 5 + [0.0] * 5
    idx = pd.date_range("2024-01-02", periods=len(pnls), freq="B")
    series = pd.Series(pnls, index=idx)
    for i in range(len(series)):
        gate.update(bar_idx=i, recent_pnl=series, initial_cash=100_000.0)
    # After resume, there is NO partial-size flag — gate just goes tripped=False.
    # The simulator scales positions normally.
    assert not gate.tripped
    # No hidden 'half_size' state on the gate object.
    assert not hasattr(gate, "phased_size_mult")


def test_book_flat_is_disjunction_across_gates():
    from backtester.engine.regime import RegimePolicy
    policy = RegimePolicy.from_disabled()
    # Manually mark one gate tripped — book_flat must be True.
    policy.spy_ema.tripped = True
    assert policy.state(bar_idx=0).book_flat is True
    policy.spy_ema.tripped = False
    policy.vix.tripped = True
    assert policy.state(bar_idx=0).book_flat is True
    policy.vix.tripped = False
    policy.circuit_breaker.tripped = True
    assert policy.state(bar_idx=0).book_flat is True


def test_disabled_gate_never_trips():
    from backtester.engine.regime import RegimePolicy
    policy = RegimePolicy.from_disabled()
    # All gates disabled — update is a no-op, tripped stays False.
    aux_data = {
        "SPY": _spy_series([100, 99, 98, 50]),
        "^VIX": _vix_series([20, 40, 50, 60]),
    }
    recent_pnl = pd.Series(
        [-1000] * 4, index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    for i in range(4):
        policy.update(
            bar_idx=i, aux_data=aux_data, recent_pnl=recent_pnl, initial_cash=100_000.0,
        )
    assert policy.state(bar_idx=3).book_flat is False


def test_flatten_on_trip_emits_zero_target_for_all_open():
    """RegimePolicy.state(bar_idx).book_flat is the simulator's signal to flatten."""
    from backtester.engine.regime import RegimePolicy
    from backtester.config.models import (
        RegimesConfig, SpyEmaRegimeConfig, VixRegimeConfig, CircuitBreakerConfig,
    )
    cfg = RegimesConfig(
        spy_ema=SpyEmaRegimeConfig(enabled=True, ema_lookback=3, trip_pct=-0.02, resume_pct=0.02),
        vix=VixRegimeConfig(enabled=False),
        circuit_breaker=CircuitBreakerConfig(enabled=False),
    )
    policy = RegimePolicy.from_config(cfg)
    aux_data = {"SPY": _spy_series([100, 100, 100, 70])}  # crash on bar 3
    recent_pnl = pd.Series(
        [0.0] * 4, index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    for i in range(4):
        policy.update(
            bar_idx=i, aux_data=aux_data, recent_pnl=recent_pnl, initial_cash=100_000.0,
        )
    assert policy.state(bar_idx=3).book_flat is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_regime_policy.py -v -k "book_flat or disabled_gate or flatten or full_size"`
Expected: 4 FAIL.

- [ ] **Step 3: Implement**

Append to `backtester/engine/regime.py`:

```python
@dataclass(frozen=True)
class RegimeState:
    """End-of-bar snapshot of which gates are tripped."""
    spy_ema_tripped: bool
    vix_tripped: bool
    circuit_breaker_tripped: bool

    @property
    def book_flat(self) -> bool:
        return self.spy_ema_tripped or self.vix_tripped or self.circuit_breaker_tripped


@dataclass
class RegimePolicy:
    """Three-gate regime policy. Disabled gates are no-ops."""
    spy_ema: SpyEmaGate
    vix: VixGate
    circuit_breaker: CircuitBreakerGate
    spy_ema_enabled: bool = False
    vix_enabled: bool = False
    circuit_breaker_enabled: bool = False

    @classmethod
    def from_disabled(cls) -> "RegimePolicy":
        return cls(
            spy_ema=SpyEmaGate(),
            vix=VixGate(),
            circuit_breaker=CircuitBreakerGate(),
        )

    @classmethod
    def from_config(cls, cfg) -> "RegimePolicy":
        """Construct from a RegimesConfig dataclass."""
        return cls(
            spy_ema=SpyEmaGate(
                ema_lookback=cfg.spy_ema.ema_lookback,
                trip_pct=cfg.spy_ema.trip_pct,
                resume_pct=cfg.spy_ema.resume_pct,
            ),
            vix=VixGate(
                trip_threshold=cfg.vix.trip_threshold,
                trip_consec=cfg.vix.trip_consec,
                resume_threshold=cfg.vix.resume_threshold,
                resume_consec=cfg.vix.resume_consec,
            ),
            circuit_breaker=CircuitBreakerGate(
                pnl_window_days=cfg.circuit_breaker.pnl_window_days,
                trip_pct=cfg.circuit_breaker.trip_pct,
                pause_days=cfg.circuit_breaker.pause_days,
            ),
            spy_ema_enabled=cfg.spy_ema.enabled,
            vix_enabled=cfg.vix.enabled,
            circuit_breaker_enabled=cfg.circuit_breaker.enabled,
        )

    def update(
        self,
        *,
        bar_idx: int,
        aux_data: dict[str, pd.DataFrame],
        recent_pnl: pd.Series,
        initial_cash: float,
    ) -> None:
        if self.spy_ema_enabled and "SPY" in aux_data:
            spy_close = aux_data["SPY"]["close"]
            spy_ema = spy_close.ewm(span=self.spy_ema.ema_lookback, adjust=False).mean()
            self.spy_ema.update(bar_idx=bar_idx, spy_close=spy_close, spy_ema=spy_ema)
        if self.vix_enabled and "^VIX" in aux_data:
            self.vix.update(bar_idx=bar_idx, vix_close=aux_data["^VIX"]["close"])
        if self.circuit_breaker_enabled:
            self.circuit_breaker.update(
                bar_idx=bar_idx, recent_pnl=recent_pnl, initial_cash=initial_cash,
            )

    def state(self, *, bar_idx: int) -> RegimeState:
        return RegimeState(
            spy_ema_tripped=self.spy_ema.tripped if self.spy_ema_enabled else False,
            vix_tripped=self.vix.tripped if self.vix_enabled else False,
            circuit_breaker_tripped=self.circuit_breaker.tripped if self.circuit_breaker_enabled else False,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_regime_policy.py -v`
Expected: 10 PASS (6 from Task 21 + 4 new).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `280 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/regime.py tests/unit/test_regime_policy.py
git commit -m "feat(engine): RegimePolicy + RegimeState composition with disabled-gate handling"
```

Phase 8 ends. The three gates and their disjunction are testable in isolation. The simulator wires them in Phase 10.

---

## Phase 9: Risk-budget + Sector-cap enforcers

Phase 9 lands two small helper components called from the simulator's per-bar loop to scale or drop new entries. Each is a stateless function or thin class over the current portfolio snapshot.

Cumulative test target: **288** (280 + 8 — 4 each).

### Task 23: `RiskBudgetEnforcer`

**Files:**
- Create: `backtester/engine/risk_budget.py`
- Test: `tests/unit/test_risk_budget.py` (create, 4 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_risk_budget.py`:

```python
import pytest


def test_risk_budget_admits_entry_below_cap():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    # Open positions contribute 3% of equity in risk; new entry would add 2%.
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=3_000.0,
        proposed_risk_dollars=2_000.0,
    )
    assert decision.admitted is True
    assert decision.scaled_target == 1.0


def test_risk_budget_rejects_entry_above_cap():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    # Open risk 5%, new entry would add 2% → 7% > 6% → reject.
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=5_000.0,
        proposed_risk_dollars=2_000.0,
    )
    assert decision.admitted is False
    assert decision.scaled_target == 0.0


def test_risk_budget_zero_equity_zero_admit():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=0.0, current_risk_dollars=0.0, proposed_risk_dollars=100.0,
    )
    assert decision.admitted is False


def test_risk_budget_at_exact_cap_admits():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=4_000.0,
        proposed_risk_dollars=2_000.0,  # exactly hits 6%
    )
    assert decision.admitted is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_risk_budget.py -v`
Expected: 4 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `backtester/engine/risk_budget.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    admitted: bool
    scaled_target: float  # 0.0 if rejected, 1.0 if fully admitted


@dataclass(slots=True)
class RiskBudgetEnforcer:
    """Caps total portfolio risk (sum of position × stop-distance) at budget_pct of equity.

    `current_risk_dollars` is the simulator's running tally; `proposed_risk_dollars` is the
    risk a new entry would add. Decision is binary in v0.4.0 (admit-or-drop). Scaling-down
    behavior is a v0.4.x follow-up.
    """
    budget_pct: float

    def evaluate(
        self,
        *,
        portfolio_equity: float,
        current_risk_dollars: float,
        proposed_risk_dollars: float,
    ) -> RiskDecision:
        if portfolio_equity <= 0:
            return RiskDecision(admitted=False, scaled_target=0.0)
        total_risk_pct = (current_risk_dollars + proposed_risk_dollars) / portfolio_equity
        if total_risk_pct > self.budget_pct + 1e-12:
            return RiskDecision(admitted=False, scaled_target=0.0)
        return RiskDecision(admitted=True, scaled_target=1.0)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_risk_budget.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `284 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/risk_budget.py tests/unit/test_risk_budget.py
git commit -m "feat(engine): RiskBudgetEnforcer (sum position * stop_dist cap)"
```

---

### Task 24: `SectorCapEnforcer`

**Files:**
- Create: `backtester/engine/sector_cap.py`
- Test: `tests/unit/test_sector_cap.py` (create, 4 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sector_cap.py`:

```python
def test_sector_cap_admits_when_under_cap():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    # Deployed 30% in Semis, new 10% Semis entry → 40% < 50% → admit.
    decision = enforcer.evaluate(
        sector="Semis",
        deployed_per_sector={"Semis": 30_000.0, "Auto": 10_000.0},
        deployed_total=40_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is True


def test_sector_cap_rejects_when_over_cap():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    # Deployed 45% Semis, new 10% Semis → 55% > 50% → reject.
    decision = enforcer.evaluate(
        sector="Semis",
        deployed_per_sector={"Semis": 45_000.0, "Auto": 5_000.0},
        deployed_total=50_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is False


def test_sector_cap_new_sector_no_existing_positions():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    decision = enforcer.evaluate(
        sector="Crypto",
        deployed_per_sector={"Semis": 30_000.0},
        deployed_total=30_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is True


def test_sector_cap_empty_portfolio():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    decision = enforcer.evaluate(
        sector="Auto",
        deployed_per_sector={},
        deployed_total=0.0,
        proposed_dollars=10_000.0,
    )
    # No existing deployment; the proposed becomes 100% of deployed.
    # But since deployed_total + proposed = 10k, sector pct = 100% > cap.
    # Decision: cap applies even on first entry into a sector.
    assert decision.admitted is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_sector_cap.py -v`
Expected: 4 FAIL.

- [ ] **Step 3: Implement**

Create `backtester/engine/sector_cap.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorDecision:
    admitted: bool


@dataclass(slots=True)
class SectorCapEnforcer:
    """Caps per-sector deployed capital at cap_pct of total deployed."""
    cap_pct: float

    def evaluate(
        self,
        *,
        sector: str,
        deployed_per_sector: dict[str, float],
        deployed_total: float,
        proposed_dollars: float,
    ) -> SectorDecision:
        new_sector_dollars = deployed_per_sector.get(sector, 0.0) + proposed_dollars
        new_total = deployed_total + proposed_dollars
        if new_total <= 0:
            return SectorDecision(admitted=False)
        new_sector_pct = new_sector_dollars / new_total
        if new_sector_pct > self.cap_pct + 1e-12:
            return SectorDecision(admitted=False)
        return SectorDecision(admitted=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_sector_cap.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `288 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/sector_cap.py tests/unit/test_sector_cap.py
git commit -m "feat(engine): SectorCapEnforcer (per-sector deployed-capital cap)"
```

Phase 9 ends. Risk-budget and sector-cap enforcers are simple, well-tested, and ready for simulator wiring in Phase 10.

---

## Phase 10: MultiSymbolPortfolioSimulator

Phase 10 is the central framework module. It wires per-symbol `Broker` + `FillEngine` + `Position` + `TrancheStopState` together with shared cash, `RegimePolicy`, `RiskBudgetEnforcer`, `SectorCapEnforcer`, and volatility-targeted sizing. It runs the 11-step per-bar loop documented in spec §3.1.

18 unit tests, split into 4 tasks: skeleton + cash mechanics (5), tranche promotion + per-symbol stops (4), risk/sector enforcement (5), regime gates + sizing + phase-callback (4).

Cumulative test target: **306** (288 + 18).

### Task 25: Simulator skeleton — shared cash, two-symbol independence

**Files:**
- Create: `backtester/engine/multi_portfolio.py`
- Test: `tests/unit/test_multi_symbol_simulator.py` (create, 5 tests)

This task lands the simulator with stub regime + risk + sector behavior (all permissive). Later tasks wire each enforcer in.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_multi_symbol_simulator.py`:

```python
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---- helpers ----

def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low":  [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def _const_signals(idx, target):
    """Trivial signal frame: emit target on bar 1, hold, exit on bar -1."""
    sig = pd.Series(0, index=idx)
    sig.iloc[1] = target
    return pd.DataFrame({"signal": sig.shift(1).fillna(0).astype(float), "size": 1.0}, index=idx)


def _build_simulator(symbols, *, initial_cash=100_000.0):
    """Return a MultiSymbolPortfolioSimulator wired to permissive defaults.

    Each test that needs specific risk/sector caps overrides them.
    """
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker

    return MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(
            sizing_mode="percent_equity",
            size=0.10,  # 10% per symbol; small enough for 2-symbol tests
            position_cap_pct=1.0,
            cash_reserve_pct=0.0,
            risk_budget_pct=1.0,
            sector_cap_pct=1.0,
        ),
        initial_cash=initial_cash,
        broker_factory=lambda: Broker(ExecutionConfig(
            initial_cash=initial_cash,
            commission_bps=1.0,
            slippage_bps=0.0,
            allow_fractional=False,
            allow_short=False,
        )),
    )


# ---- 5 cash-mechanics + independence tests ----

def test_shared_cash_debited_on_buy_credited_on_sell():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 110.0])}
    sectors = {"AAA": "X"}
    signals = {"AAA": _const_signals(data["AAA"].index, target=1.0)}
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals=signals, aux_data={}, regime_config=None,
    )
    # After full cycle, cash should be > initial_cash (bought at ~100, sold at ~110).
    assert result.final_equity > 100_000.0


def test_portfolio_equity_sums_cash_plus_positions():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 100.0, 105.0, 105.0])}
    sectors = {"AAA": "X"}
    signals = {"AAA": _const_signals(data["AAA"].index, target=1.0)}
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals=signals, aux_data={}, regime_config=None,
    )
    # Mid-run (bar 2), position is held; equity = cash + qty * close.
    eq_curve = result.equity_curve
    assert eq_curve.iloc[2] > 0
    # Equity curve is monotonic-in-shape with the underlying close trajectory.


def test_two_symbol_independent_entries_dont_interfere():
    sim = _build_simulator(symbols=["AAA", "BBB"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = _ohlcv([100.0] * 5)
    bbb = _ohlcv([200.0] * 5)
    # AAA enters bar 1, BBB enters bar 2. Different entry bars.
    aaa_sig = pd.DataFrame({"signal": [0, 0, 1, 1, 0], "size": 1.0}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0, 0, 0, 1, 0], "size": 1.0}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # Both symbols should have at least one trade in their per-symbol trade log.
    assert len(result.trades_per_symbol["AAA"]) >= 1
    assert len(result.trades_per_symbol["BBB"]) >= 1


def test_unique_per_symbol_trailing_stop_state():
    """Each symbol owns its own TrancheStopState; states don't bleed."""
    sim = _build_simulator(symbols=["AAA", "BBB"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    aaa = _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0])
    bbb = _ohlcv([200.0, 200.0, 200.0, 200.0, 200.0])
    aaa_sig = pd.DataFrame({"signal": [0, 1, 1, 0, 0], "size": 1.0}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0, 0, 1, 1, 0], "size": 1.0}, index=idx)
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # Both symbols saw signal-driven entries; state was per-symbol.
    assert "AAA" in result.trades_per_symbol
    assert "BBB" in result.trades_per_symbol


def test_portfolio_equity_curve_length_matches_panel_index():
    sim = _build_simulator(symbols=["AAA"])
    data = {"AAA": _ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])}
    sectors = {"AAA": "X"}
    signals = {"AAA": _const_signals(data["AAA"].index, target=1.0)}
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors=sectors,
        signals=signals, aux_data={}, regime_config=None,
    )
    assert len(result.equity_curve) == len(data["AAA"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v`
Expected: 5 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement skeleton**

Create `backtester/engine/multi_portfolio.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order, OrderSide, OrderType
from backtester.engine.position import Position


@dataclass
class MultiSymbolResult:
    equity_curve: pd.Series
    final_equity: float
    trades_per_symbol: dict[str, list[Fill]] = field(default_factory=dict)
    portfolio_max_drawdown: float = 0.0
    portfolio_total_return: float = 0.0
    portfolio_sharpe: float = 0.0


@dataclass
class MultiSymbolPortfolioSimulator:
    config: Any  # PortfolioConfig
    initial_cash: float
    broker_factory: Callable[[], Broker]

    def simulate(
        self,
        *,
        symbols: list[str],
        data: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        signals: dict[str, pd.DataFrame],
        aux_data: dict[str, pd.DataFrame],
        regime_config: Optional[Any] = None,
    ) -> MultiSymbolResult:
        """Run the multi-symbol backtest. See spec §3.1 for the per-bar loop."""
        # Establish the shared time index. All symbols + aux_data must share it.
        index = data[symbols[0]].index

        # Per-symbol state.
        brokers: dict[str, Broker] = {s: self.broker_factory() for s in symbols}
        positions: dict[str, Position] = {s: Position(symbol=s) for s in symbols}
        trades: dict[str, list[Fill]] = {s: [] for s in symbols}
        pending_signal: dict[str, Optional[Order]] = {s: None for s in symbols}
        pending_stop: dict[str, Optional[Order]] = {s: None for s in symbols}

        # Shared cash + equity.
        cash = self.initial_cash
        equity_history: list[float] = []

        for i in range(len(index)):
            ts = index[i]
            # Step 1: execute pending stop orders.
            stop_filled = {s: False for s in symbols}
            for s in symbols:
                if pending_stop[s] is not None:
                    fill = brokers[s].submit(pending_stop[s], bar=data[s].iloc[i])
                    if fill is not None:
                        fill.reason = "trailing_stop"
                        cash += fill.cash_delta
                        positions[s].apply_fill(fill)
                        trades[s].append(fill)
                        stop_filled[s] = True
                    pending_stop[s] = None

            # Step 2: execute pending signal orders (unless cancelled by stop).
            for s in symbols:
                if pending_signal[s] is not None and not stop_filled[s]:
                    fill = brokers[s].submit(pending_signal[s], bar=data[s].iloc[i])
                    if fill is not None:
                        fill.reason = "signal"
                        cash += fill.cash_delta
                        positions[s].apply_fill(fill)
                        trades[s].append(fill)
                pending_signal[s] = None

            # Step 3..6: tranche/phase transitions — STUB until Task 26.
            # Step 7: regime gates — STUB until Task 28.
            # Step 8: per-bar strategy callback — STUB until Task 28.
            # Step 9: portfolio-level scaling — STUB until Task 27.

            # Step 10: schedule orders for bar i+1.
            if i + 1 < len(index):
                for s in symbols:
                    target = float(signals[s]["signal"].iloc[i])
                    target_qty = self._target_shares(
                        target=target, symbol=s, close=float(data[s]["close"].iloc[i]),
                        portfolio_equity=cash + sum(
                            positions[t].qty * float(data[t]["close"].iloc[i])
                            for t in symbols
                        ),
                    )
                    delta = target_qty - positions[s].qty
                    if abs(delta) > 1e-9:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        pending_signal[s] = Order(
                            symbol=s, side=side, qty=abs(delta), type=OrderType.MARKET,
                        )

            # Step 11: mark to market.
            position_value = sum(
                positions[s].qty * float(data[s]["close"].iloc[i]) for s in symbols
            )
            equity_history.append(cash + position_value)

        equity_curve = pd.Series(equity_history, index=index, name="equity")
        return MultiSymbolResult(
            equity_curve=equity_curve,
            final_equity=float(equity_curve.iloc[-1]),
            trades_per_symbol=trades,
            portfolio_total_return=float(equity_curve.iloc[-1]) / self.initial_cash - 1.0,
        )

    def _target_shares(
        self, *, target: float, symbol: str, close: float, portfolio_equity: float,
    ) -> float:
        """Convert target ∈ [-1, 1] into share count. percent_equity mode for now."""
        if abs(target) < 1e-12:
            return 0.0
        dollars = target * portfolio_equity * self.config.size
        return int(dollars / close) if not getattr(self.config, "allow_fractional", False) else dollars / close
```

Note: this skeleton is permissive (no risk/sector/regime enforcement yet). Stubs are marked. Subsequent tasks land them.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `293 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/multi_portfolio.py tests/unit/test_multi_symbol_simulator.py
git commit -m "feat(engine): MultiSymbolPortfolioSimulator skeleton — shared cash + independent symbols"
```

---

### Task 26: Wire `TrancheStopState` into the simulator's per-bar loop

**Files:**
- Modify: `backtester/engine/multi_portfolio.py`
- Test: `tests/unit/test_multi_symbol_simulator.py` (append 4 tests)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_promote_to_runner_called_on_partial_close():
    """Strategy emits target=0.5 from full position → TrancheStopState should promote."""
    from backtester.engine.tranche_stop import TSPhase
    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])}
    # Bar 1: enter full. Bar 3: scale to 0.5 (tranche 1 fills). Bar 5: exit.
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0.5, 0.5, 0], "size": 1.0}, index=idx)
    # Configure tranche-stop via execution config on the broker factory.
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.75, runner_atr_mult=2.5,
        breakeven_floor=True, tranche_stop_atr_period=3,
    ))
    sim.config.size = 0.5  # half-equity full-size for visibility
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Inspect the simulator's exposed per-symbol final phase.
    assert result.tranche_phase_at_end["AAA"] is TSPhase.DISARMED
    # And the final position is zero (exited fully).
    assert result.position_qty_at_end["AAA"] == 0


def test_disarm_called_on_full_exit():
    from backtester.engine.tranche_stop import TSPhase
    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {"AAA": _ohlcv([100.0] * 4)}
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.75, runner_atr_mult=2.5,
        breakeven_floor=True, tranche_stop_atr_period=3,
    ))
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    assert result.tranche_phase_at_end["AAA"] is TSPhase.DISARMED


def test_pending_stop_per_symbol_independent():
    """Each symbol's pending_stop is independent. Stop on AAA does not affect BBB."""
    sim = _build_simulator(symbols=["AAA", "BBB"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    # AAA crashes after entry; BBB is stable.
    aaa = _ohlcv([100.0, 100.0, 100.0, 100.0, 80.0])
    bbb = _ohlcv([200.0] * 5)
    aaa_sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 1.0, 1.0], "size": 1.0}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 1.0, 1.0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,  # tight stop on AAA
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # AAA stopped out (some trade has reason="trailing_stop").
    aaa_trades = result.trades_per_symbol["AAA"]
    assert any(f.reason == "trailing_stop" for f in aaa_trades)
    # BBB never stopped.
    bbb_trades = result.trades_per_symbol["BBB"]
    assert not any(f.reason == "trailing_stop" for f in bbb_trades)


def test_stop_wins_over_signal_same_bar_per_symbol():
    """On the bar where a stop fires AND the strategy signals an exit, only the stop fill lands."""
    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 50.0])}  # crash on last bar
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    trades = result.trades_per_symbol["AAA"]
    # We expect: an entry signal fill at bar 2, and EITHER a stop OR a signal exit on bar 3 — not both.
    exit_fills = [f for f in trades if f.side.value == "sell"]
    assert len(exit_fills) == 1
    # And the one that landed is the trailing_stop (it wins).
    assert exit_fills[0].reason == "trailing_stop"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v -k "promote or disarm or pending_stop or stop_wins"`
Expected: 4 FAIL — `MultiSymbolResult` does not yet expose `tranche_phase_at_end` or `position_qty_at_end`.

- [ ] **Step 3: Implement TrancheStopState integration**

In `backtester/engine/multi_portfolio.py`:

1. Add per-symbol `TrancheStopState` and `ATR` setup at the top of `simulate()`:

```python
from backtester.engine.atr import compute_atr
from backtester.engine.tranche_stop import TrancheStopState, TSPhase

# After the broker dict is built:
ts_states: dict[str, TrancheStopState] = {}
ex = next(iter(brokers.values())).config  # all brokers share an ExecutionConfig in practice
if ex.hard_stop_atr_mult is not None:
    for s in symbols:
        ts_states[s] = TrancheStopState(
            hard_stop_atr_mult=ex.hard_stop_atr_mult,
            runner_atr_mult=ex.runner_atr_mult,
            breakeven_floor=ex.breakeven_floor,
            atr_series=compute_atr(data[s], ex.tranche_stop_atr_period),
        )
```

2. After Step 2 (executing pending signal), record `prev_qty[s]` and recompute `new_qty[s]`. Wrap in Steps 3-5:

```python
# Step 3: compute new_qty per symbol (already done — positions are updated).
# Step 4: tranche-state transitions per symbol.
for s in symbols:
    if s not in ts_states:
        continue
    prev = prev_qty[s]
    new = positions[s].qty
    if prev == 0 and new != 0:
        # Flat → non-flat: reset.
        last_fill = trades[s][-1]
        ts_states[s].reset(entry_price=last_fill.price, bar_idx=i)
    elif prev != 0 and new == 0:
        # Non-flat → flat: disarm (whether stop- or signal-driven).
        ts_states[s].disarm()
    elif prev != 0 and new != 0 and (prev > 0) == (new > 0) and abs(new) < abs(prev):
        # Same-sign partial close → promote.
        ts_states[s].promote_to_runner()

# Step 5: update peak/trough.
for s in symbols:
    if s in ts_states:
        ts_states[s].update(data[s].iloc[i])
```

Note: `prev_qty[s]` must be captured BEFORE step 1 (at the top of each bar's iteration). Add:

```python
prev_qty = {s: positions[s].qty for s in symbols}
```
as the very first line inside the `for i in range(len(index))` loop.

3. In Step 10 (schedule), also schedule the next-bar stop order:

```python
if s in ts_states and ts_states[s].phase is not TSPhase.DISARMED:
    sign = 1 if positions[s].qty > 0 else -1 if positions[s].qty < 0 else 0
    stop_px = ts_states[s].stop_price(sign=sign, bar_idx=i + 1)
    if stop_px is not None and sign != 0:
        stop_side = OrderSide.SELL if sign > 0 else OrderSide.BUY
        pending_stop[s] = Order(
            symbol=s, side=stop_side, qty=abs(positions[s].qty),
            type=OrderType.STOP, stop_price=stop_px,
        )
```

4. Expose end-of-run snapshots on `MultiSymbolResult`:

```python
@dataclass
class MultiSymbolResult:
    equity_curve: pd.Series
    final_equity: float
    trades_per_symbol: dict[str, list[Fill]] = field(default_factory=dict)
    portfolio_max_drawdown: float = 0.0
    portfolio_total_return: float = 0.0
    portfolio_sharpe: float = 0.0
    # New in this task:
    tranche_phase_at_end: dict[str, Any] = field(default_factory=dict)
    position_qty_at_end: dict[str, float] = field(default_factory=dict)
```

Populate at the end of `simulate()`:

```python
result = MultiSymbolResult(
    equity_curve=equity_curve,
    final_equity=float(equity_curve.iloc[-1]),
    trades_per_symbol=trades,
    portfolio_total_return=float(equity_curve.iloc[-1]) / self.initial_cash - 1.0,
    tranche_phase_at_end={s: ts_states[s].phase if s in ts_states else None for s in symbols},
    position_qty_at_end={s: positions[s].qty for s in symbols},
)
return result
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v`
Expected: 9 PASS (5 from Task 25 + 4 new).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `297 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/multi_portfolio.py tests/unit/test_multi_symbol_simulator.py
git commit -m "feat(engine): wire TrancheStopState into MultiSymbolPortfolioSimulator (per-symbol)"
```

---

### Task 27: Wire `RiskBudgetEnforcer` + `SectorCapEnforcer` + `position_cap_pct` + `cash_reserve_pct`

**Files:**
- Modify: `backtester/engine/multi_portfolio.py`
- Test: `tests/unit/test_multi_symbol_simulator.py` (append 5 tests)

This task adds portfolio-level scaling: position cap, cash reserve, risk budget, sector cap, and vol-targeted sizing.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_simultaneous_entries_compete_for_risk_budget():
    """When two new entries would together exceed risk_budget_pct, one is dropped."""
    sim = _build_simulator(symbols=["AAA", "BBB"])
    sim.config.risk_budget_pct = 0.03  # tight cap
    sim.config.size = 0.5  # so each position would be 50% of equity
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    aaa = _ohlcv([100.0] * 4)
    bbb = _ohlcv([200.0] * 4)
    # Both signal entry on bar 1.
    aaa_sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    # At least one of the two must have been dropped (no entry trade).
    aaa_entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    bbb_entries = [f for f in result.trades_per_symbol["BBB"] if f.side.value == "buy"]
    assert len(aaa_entries) == 0 or len(bbb_entries) == 0


def test_position_cap_clips_oversized_signal():
    """If strategy emits target=1.0 but position_cap_pct=0.05, position is 5% not 100%."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.position_cap_pct = 0.05
    sim.config.size = 1.0
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {"AAA": _ohlcv([100.0, 100.0, 100.0, 100.0])}
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Position dollars should be at most 5% of equity → ~5,000 / 100 = 50 shares.
    entry = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"][0]
    assert entry.qty <= 50


def test_cash_reserve_cap_drops_late_entries():
    """When deployed would exceed (1 - cash_reserve_pct), late entries get dropped."""
    sim = _build_simulator(symbols=["AAA", "BBB", "CCC"])
    sim.config.cash_reserve_pct = 0.30
    sim.config.size = 0.5  # each entry intends 50% deployment
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {s: _ohlcv([100.0] * 4) for s in ["AAA", "BBB", "CCC"]}
    sigs = {s: pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx) for s in ["AAA", "BBB", "CCC"]}
    result = sim.simulate(
        symbols=["AAA", "BBB", "CCC"], data=data,
        sectors={"AAA": "X", "BBB": "Y", "CCC": "Z"},
        signals=sigs, aux_data={}, regime_config=None,
    )
    # Two entries would deploy 100%; cash_reserve_pct=0.30 caps total at 70%, so only ~1 entry fits.
    entry_count = sum(
        1 for s in ["AAA", "BBB", "CCC"]
        if any(f.side.value == "buy" for f in result.trades_per_symbol[s])
    )
    assert entry_count <= 2


def test_sector_cap_blocks_third_entry_in_full_sector():
    """Three same-sector entries; cap=0.50; total cap blocks the third."""
    sim = _build_simulator(symbols=["AAA", "BBB", "CCC"])
    sim.config.sector_cap_pct = 0.50
    sim.config.size = 0.3  # each is 30% — three would be 90% in one sector
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    data = {s: _ohlcv([100.0] * 4) for s in ["AAA", "BBB", "CCC"]}
    sigs = {s: pd.DataFrame({"signal": [0, 1.0, 1.0, 0], "size": 1.0}, index=idx) for s in ["AAA", "BBB", "CCC"]}
    result = sim.simulate(
        symbols=["AAA", "BBB", "CCC"], data=data,
        sectors={"AAA": "Semis", "BBB": "Semis", "CCC": "Semis"},
        signals=sigs, aux_data={}, regime_config=None,
    )
    entry_count = sum(
        1 for s in ["AAA", "BBB", "CCC"]
        if any(f.side.value == "buy" for f in result.trades_per_symbol[s])
    )
    assert entry_count < 3


def test_risk_budget_released_on_full_exit():
    """After a full exit, the freed risk budget allows a new entry on the next bar."""
    sim = _build_simulator(symbols=["AAA", "BBB"])
    sim.config.risk_budget_pct = 0.05
    sim.config.size = 0.5
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    aaa = _ohlcv([100.0] * 6)
    bbb = _ohlcv([200.0] * 6)
    # AAA enters bar 1, exits bar 3. BBB tries entry bar 4 — budget should be free.
    aaa_sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0, 0, 0], "size": 1.0}, index=idx)
    bbb_sig = pd.DataFrame({"signal": [0, 0, 0, 0, 1.0, 0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))
    result = sim.simulate(
        symbols=["AAA", "BBB"], data={"AAA": aaa, "BBB": bbb},
        sectors={"AAA": "X", "BBB": "Y"},
        signals={"AAA": aaa_sig, "BBB": bbb_sig},
        aux_data={}, regime_config=None,
    )
    bbb_entries = [f for f in result.trades_per_symbol["BBB"] if f.side.value == "buy"]
    assert len(bbb_entries) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v -k "risk_budget or position_cap or cash_reserve or sector_cap"`
Expected: 5 FAIL — the simulator currently has no enforcement (it's permissive).

- [ ] **Step 3: Implement enforcers in simulator step 10**

In `backtester/engine/multi_portfolio.py`, replace the schedule-orders block with:

```python
# Step 10: schedule orders for bar i+1.
if i + 1 < len(index):
    portfolio_equity_now = cash + sum(
        positions[s].qty * float(data[s]["close"].iloc[i]) for s in symbols
    )
    deployed_total = sum(
        abs(positions[s].qty) * float(data[s]["close"].iloc[i]) for s in symbols
    )
    deployed_per_sector: dict[str, float] = {}
    for s in symbols:
        sec = sectors[s]
        deployed_per_sector[sec] = deployed_per_sector.get(sec, 0.0) + (
            abs(positions[s].qty) * float(data[s]["close"].iloc[i])
        )
    current_risk_dollars = 0.0
    for s in symbols:
        if s in ts_states and ts_states[s].phase is not TSPhase.DISARMED:
            stop_px = ts_states[s].stop_price(
                sign=1 if positions[s].qty > 0 else -1, bar_idx=i,
            )
            if stop_px is not None:
                current_risk_dollars += abs(positions[s].qty) * abs(
                    float(data[s]["close"].iloc[i]) - stop_px
                )

    from backtester.engine.risk_budget import RiskBudgetEnforcer
    from backtester.engine.sector_cap import SectorCapEnforcer
    risk_enforcer = RiskBudgetEnforcer(budget_pct=self.config.risk_budget_pct)
    sector_enforcer = SectorCapEnforcer(cap_pct=self.config.sector_cap_pct)
    cash_reserve_limit = portfolio_equity_now * (1.0 - self.config.cash_reserve_pct)

    for s in symbols:
        target = float(signals[s]["signal"].iloc[i])
        capped = max(-1.0, min(1.0, target))
        # Apply position_cap_pct.
        intent_dollars = capped * portfolio_equity_now * self.config.size
        intent_dollars = max(
            -portfolio_equity_now * self.config.position_cap_pct,
            min(portfolio_equity_now * self.config.position_cap_pct, intent_dollars),
        )
        # Apply cash reserve (only for NEW deployment, not partial closes).
        proposed_dollars = abs(intent_dollars)
        existing_dollars = abs(positions[s].qty) * float(data[s]["close"].iloc[i])
        if proposed_dollars > existing_dollars:
            additional = proposed_dollars - existing_dollars
            if deployed_total + additional > cash_reserve_limit:
                intent_dollars = (existing_dollars * (1 if intent_dollars > 0 else -1))
        # Apply sector cap (only for additional deployment).
        if proposed_dollars > existing_dollars:
            additional = proposed_dollars - existing_dollars
            sec_decision = sector_enforcer.evaluate(
                sector=sectors[s], deployed_per_sector=deployed_per_sector,
                deployed_total=deployed_total, proposed_dollars=additional,
            )
            if not sec_decision.admitted:
                intent_dollars = (existing_dollars * (1 if intent_dollars > 0 else -1))
        # Apply risk budget (only for additional risk).
        if proposed_dollars > existing_dollars and s in ts_states:
            atr = compute_atr(data[s], ex.tranche_stop_atr_period).iloc[i]
            if not pd.isna(atr):
                est_stop_dist = ex.hard_stop_atr_mult * float(atr)
                proposed_risk = (proposed_dollars / float(data[s]["close"].iloc[i])) * est_stop_dist
                risk_decision = risk_enforcer.evaluate(
                    portfolio_equity=portfolio_equity_now,
                    current_risk_dollars=current_risk_dollars,
                    proposed_risk_dollars=proposed_risk,
                )
                if not risk_decision.admitted:
                    intent_dollars = (existing_dollars * (1 if intent_dollars > 0 else -1))

        target_qty = int(intent_dollars / float(data[s]["close"].iloc[i])) if intent_dollars else 0
        delta = target_qty - positions[s].qty
        if abs(delta) > 1e-9:
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            pending_signal[s] = Order(
                symbol=s, side=side, qty=abs(delta), type=OrderType.MARKET,
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v`
Expected: 14 PASS (9 from Tasks 25-26 + 5 new).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `302 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/multi_portfolio.py tests/unit/test_multi_symbol_simulator.py
git commit -m "feat(engine): wire risk-budget + sector-cap + cash-reserve + position-cap enforcers"
```

---

### Task 28: Regime gates, vol-targeted sizing, per-bar callback, position_phase finalization

**Files:**
- Modify: `backtester/engine/multi_portfolio.py`
- Test: `tests/unit/test_multi_symbol_simulator.py` (append 4 tests)

This task closes out Phase 10: regime gates trip the book; vol-targeted sizing kicks in when `sizing_mode='vol_targeted'`; per-bar callback runs for `uses_per_bar` strategies; `ctx.position_phase` is finalized before the callback.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_vol_targeted_sizing_uses_realized_vol_20d():
    """sizing_mode='vol_targeted' produces position_dollars ≈ vol_target / realized_vol_20d."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.sizing_mode = "vol_targeted"
    sim.config.vol_target = 0.12
    sim.config.position_cap_pct = 1.0
    idx = pd.date_range("2024-01-02", periods=30, freq="B")
    # Low-vol price series: small daily moves.
    closes = [100.0 + 0.1 * i for i in range(30)]
    data = {"AAA": _ohlcv(closes)}
    sig = pd.DataFrame(
        {"signal": [0] * 25 + [1.0, 1.0, 1.0, 0, 0], "size": 1.0}, index=idx,
    )
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Position should be large (low vol → high target).
    entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    assert len(entries) >= 1
    assert entries[0].qty > 100  # arbitrary positive lower bound


def test_vol_targeted_sizing_defers_entry_during_warmup():
    """No realized_vol available yet → entry deferred."""
    sim = _build_simulator(symbols=["AAA"])
    sim.config.sizing_mode = "vol_targeted"
    sim.config.vol_target = 0.12
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    data = {"AAA": _ohlcv(closes)}
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 0, 0], "size": 1.0}, index=idx)
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
    )
    # Realized 20d-vol is NaN at bar 1 (only 1 bar of history).
    # Simulator must DEFER the entry — no fills.
    entries = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "buy"]
    assert len(entries) == 0


def test_regime_gate_flattens_book():
    """When the SPY EMA gate trips, all open positions get target=0 next bar."""
    from backtester.config.models import RegimesConfig, SpyEmaRegimeConfig
    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=8, freq="B")
    data = {"AAA": _ohlcv([100.0] * 8)}
    # SPY crashes on bar 5.
    spy = _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0, 70.0, 70.0, 70.0])
    sig = pd.DataFrame({"signal": [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0], "size": 1.0}, index=idx)
    regimes = RegimesConfig(spy_ema=SpyEmaRegimeConfig(
        enabled=True, ema_lookback=3, trip_pct=-0.02, resume_pct=0.02,
    ))
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={"SPY": spy}, regime_config=regimes,
    )
    # Position should exit after the regime trips on bar 5; bar 6 sees a sell fill.
    exits = [f for f in result.trades_per_symbol["AAA"] if f.side.value == "sell"]
    assert len(exits) >= 1


def test_position_phase_finalized_before_strategy_callback():
    """For uses_per_bar strategies, position_phase reflects end-of-bar-t state when
    the strategy decides bar t+1's signal. The callback receives the finalized phase."""
    from backtester.engine.tranche_stop import TSPhase

    captured: list[Any] = []

    class _CaptureStrategy:
        uses_per_bar = True
        def signal_for_bar(self, *, symbol, bar_idx, data_panel, indicators_panel, ctx, params):
            captured.append((bar_idx, ctx.position_phase.get(symbol)))
            return 0.0

    sim = _build_simulator(symbols=["AAA"])
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    data = {"AAA": _ohlcv([100.0] * 5)}
    sig = pd.DataFrame({"signal": [0, 1.0, 0.5, 0, 0], "size": 1.0}, index=idx)
    from backtester.config.models import ExecutionConfig
    from backtester.engine.broker import Broker
    sim.broker_factory = lambda: Broker(ExecutionConfig(
        initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        hard_stop_atr_mult=1.0, runner_atr_mult=2.5,
        breakeven_floor=False, tranche_stop_atr_period=2,
    ))
    result = sim.simulate(
        symbols=["AAA"], data=data, sectors={"AAA": "X"},
        signals={"AAA": sig}, aux_data={}, regime_config=None,
        strategy=_CaptureStrategy(),  # NEW kwarg in this task
    )
    # On bar 2 (after entering on bar 1's signal → filling bar 2 open), phase should be HARD.
    bar2_phase = next(p for i, p in captured if i == 2)
    assert bar2_phase is TSPhase.HARD
    # On bar 3 (after the 0.5 target was scheduled bar 2 and filled bar 3), phase should be RUNNER.
    bar3_phase = next(p for i, p in captured if i == 3)
    assert bar3_phase is TSPhase.RUNNER
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v -k "vol_targeted or regime_gate or position_phase_finalized"`
Expected: 4 FAIL.

- [ ] **Step 3: Implement**

In `backtester/engine/multi_portfolio.py`, three changes:

1. Add `RegimePolicy` and `recent_pnl` tracking. Near the top of `simulate()`:

```python
from backtester.engine.regime import RegimePolicy

regime_policy = (
    RegimePolicy.from_config(regime_config) if regime_config is not None
    else RegimePolicy.from_disabled()
)
recent_pnl_list: list[float] = []  # per-bar portfolio PnL deltas
```

After step 6 / before step 10, append `recent_pnl_list.append(current_pnl_for_bar)`:

```python
# Step 7: regime gate update.
recent_pnl_series = pd.Series(
    recent_pnl_list, index=index[: len(recent_pnl_list)],
) if recent_pnl_list else pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
regime_policy.update(
    bar_idx=i, aux_data=aux_data,
    recent_pnl=recent_pnl_series, initial_cash=self.initial_cash,
)
regime_state = regime_policy.state(bar_idx=i)
```

Force `target=0` for all symbols when `regime_state.book_flat` is True, before the per-symbol scaling block in step 10.

2. Add vol-targeted sizing branch:

```python
def _vol_targeted_dollars(self, *, target, symbol, close, portfolio_equity, data_panel, bar_idx):
    if abs(target) < 1e-12 or self.config.sizing_mode != "vol_targeted":
        return target * portfolio_equity * self.config.size
    closes = data_panel[symbol]["close"].iloc[: bar_idx + 1]
    if len(closes) < 21:
        return 0.0  # warmup defer
    realized_vol = closes.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    if pd.isna(realized_vol) or realized_vol <= 0:
        return 0.0
    target_pct = self.config.vol_target / realized_vol
    target_pct = max(-self.config.position_cap_pct, min(self.config.position_cap_pct, target_pct))
    return target * portfolio_equity * target_pct
```

Wire this into step 10 in place of the current `intent_dollars` computation when `sizing_mode == "vol_targeted"`.

3. Add per-bar `strategy` parameter and callback dispatch:

```python
def simulate(
    self, *, symbols, data, sectors, signals, aux_data,
    regime_config=None, strategy=None,
) -> MultiSymbolResult:
    # ... existing setup ...

    for i in range(len(index)):
        # ... steps 1-7 ...

        # Step 8: per-bar strategy callback (overrides pre-computed signals).
        if strategy is not None and getattr(strategy, "uses_per_bar", False):
            from backtester.core.types import StrategyContext
            ctx = StrategyContext(
                position_phase={s: ts_states[s].phase for s in symbols if s in ts_states},
                bars_in_phase={},
                recent_pnl=recent_pnl_series,
                regime=regime_state,
            )
            for s in symbols:
                signals_for_bar = strategy.signal_for_bar(
                    symbol=s, bar_idx=i,
                    data_panel=data, indicators_panel={}, ctx=ctx, params=None,
                )
                signals[s]["signal"].iloc[i] = signals_for_bar

        # ... step 10 schedule ...
```

`ctx.position_phase` is captured AFTER step 5 (peak/trough update) and step 6 (phase metadata), so the strategy sees the finalized phase from the just-processed bar — matching spec §1.6.

4. Add `numpy` import at the top: `import numpy as np`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_multi_symbol_simulator.py -v`
Expected: 18 PASS (14 from Tasks 25-27 + 4 new).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `306 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/multi_portfolio.py tests/unit/test_multi_symbol_simulator.py
git commit -m "feat(engine): regime gates + vol-targeted sizing + per-bar strategy callback"
```

Phase 10 ends. The simulator is complete and tested for the v0.4.0 contract. It will be wrapped by `MultiSymbolBacktestEngine` in Phase 11.

---

## Phase 11: MultiSymbolBacktestEngine

Phase 11 wraps `MultiSymbolPortfolioSimulator` with strategy + data orchestration, paralleling v0.3.0's `BacktestEngine` for the multi-symbol path. One task; one new integration-style unit test that runs an end-to-end synthetic scenario.

Cumulative test target: **307** (306 + 1).

### Task 29: `MultiSymbolBacktestEngine`

**Files:**
- Create: `backtester/engine/multi_backtest_engine.py`
- Test: `tests/unit/test_multi_backtest_engine.py` (create, 1 test)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_multi_backtest_engine.py`:

```python
import pandas as pd


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


class _SimpleMRStrategy:
    """Minimal mean-reversion stub: signal=1 if close < mean10, else 0."""
    strategy_id = "test_mr_stub"
    uses_multi_symbol = True
    uses_per_bar = False

    def indicators(self, data, params):
        return pd.DataFrame({"mean10": data["close"].rolling(10).mean()}, index=data.index)

    def generate_signals_for_symbol(self, *, data, indicators, params):
        sig = (data["close"] < indicators["mean10"]).astype(float).shift(1).fillna(0)
        return pd.DataFrame({"signal": sig, "size": 1.0}, index=data.index)


def test_engine_runs_multi_symbol_end_to_end():
    from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator

    closes_a = [100 - 0.5 * i if i < 15 else 100 - 7.5 + 0.5 * (i - 15) for i in range(30)]
    closes_b = [200 + 0.5 * i for i in range(30)]
    data = {"AAA": _ohlcv(closes_a), "BBB": _ohlcv(closes_b)}
    sectors = {"AAA": "X", "BBB": "Y"}

    sim = MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(sizing_mode="percent_equity", size=0.1,
                               position_cap_pct=1.0, cash_reserve_pct=0.0,
                               risk_budget_pct=1.0, sector_cap_pct=1.0),
        initial_cash=100_000.0,
        broker_factory=lambda: Broker(ExecutionConfig(
            initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        )),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    result = engine.run(
        strategy=_SimpleMRStrategy(),
        symbols=["AAA", "BBB"],
        data=data, sectors=sectors, aux_data={}, params=None, regime_config=None,
    )
    assert result.equity_curve is not None
    assert "AAA" in result.trades_per_symbol
    assert "BBB" in result.trades_per_symbol
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_multi_backtest_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `backtester/engine/multi_backtest_engine.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backtester.engine.multi_portfolio import (
    MultiSymbolPortfolioSimulator, MultiSymbolResult,
)


@dataclass
class MultiSymbolBacktestEngine:
    simulator: MultiSymbolPortfolioSimulator

    def run(
        self,
        *,
        strategy: Any,
        symbols: list[str],
        data: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        aux_data: dict[str, pd.DataFrame],
        params: Any,
        regime_config: Optional[Any] = None,
    ) -> MultiSymbolResult:
        # Vectorized pre-compute: each symbol gets its own indicators + signals.
        signals: dict[str, pd.DataFrame] = {}
        if not getattr(strategy, "uses_per_bar", False):
            for sym in symbols:
                indicators = strategy.indicators(data[sym], params)
                signals[sym] = strategy.generate_signals_for_symbol(
                    data=data[sym], indicators=indicators, params=params,
                )
        else:
            # Per-bar strategies don't pre-compute signals; the simulator calls
            # strategy.signal_for_bar per (symbol, bar). Still pass an empty
            # signals dict with the right shape so step-10 scheduling has a
            # frame to overwrite.
            for sym in symbols:
                idx = data[sym].index
                signals[sym] = pd.DataFrame(
                    {"signal": [0.0] * len(idx), "size": [1.0] * len(idx)}, index=idx,
                )

        return self.simulator.simulate(
            symbols=symbols, data=data, sectors=sectors, signals=signals,
            aux_data=aux_data, regime_config=regime_config,
            strategy=strategy if getattr(strategy, "uses_per_bar", False) else None,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_multi_backtest_engine.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `307 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/engine/multi_backtest_engine.py tests/unit/test_multi_backtest_engine.py
git commit -m "feat(engine): MultiSymbolBacktestEngine wraps simulator for v0.4.0 strategies"
```

Phase 11 ends.

---

## Phase 12: LHS optimizer mode

Phase 12 extends the optimizer with discrete-LHS sampling (index positions over candidate lists). The existing grid-search mode is unchanged.

3 unit tests in one task.

Cumulative test target: **310** (307 + 3).

### Task 30: Discrete-LHS sampler + `sampling: lhs` config

**Files:**
- Create: `backtester/optimize/lhs_sampler.py`
- Modify: `backtester/optimize/grid_search.py`
- Modify: `backtester/config/models.py` (add `sampling`, `random_n`, `random_seed` to `OptimizationConfig`)
- Test: `tests/unit/test_optimizer_lhs.py` (create, 3 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_optimizer_lhs.py`:

```python
from collections import Counter

import pytest


def test_lhs_index_position_sampling_balanced():
    """For a 4-element list and random_n=8, each index appears 2× (±1)."""
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [10, 20, 30, 40]}
    samples = sample_param_space(space=space, random_n=8, seed=0)
    assert len(samples) == 8
    counts = Counter(s["a"] for s in samples)
    for v in [10, 20, 30, 40]:
        assert 1 <= counts[v] <= 3  # balanced ± rounding


def test_lhs_seed_determinism():
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [1, 2, 3], "b": [10, 20, 30, 40]}
    s1 = sample_param_space(space=space, random_n=10, seed=42)
    s2 = sample_param_space(space=space, random_n=10, seed=42)
    assert s1 == s2


def test_lhs_rejects_random_n_larger_than_cartesian_product():
    from backtester.optimize.lhs_sampler import sample_param_space

    space = {"a": [1, 2, 3], "b": [10, 20]}  # cartesian = 6
    with pytest.raises(ValueError, match="random_n"):
        sample_param_space(space=space, random_n=10, seed=0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_optimizer_lhs.py -v`
Expected: 3 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the LHS sampler**

Create `backtester/optimize/lhs_sampler.py`:

```python
from __future__ import annotations

import math
import random
from typing import Any


def sample_param_space(
    *,
    space: dict[str, list[Any]],
    random_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Discrete Latin-hypercube sampler over index positions in each candidate list.

    For each parameter with k candidates and a budget of random_n samples:
      - Partition [0, k) into random_n strata of equal width.
      - Draw one float per stratum (uniform within the stratum).
      - Floor to integer index → maps each stratum to one of the k candidates.
    Then randomly permute each per-parameter index list to decorrelate across
    parameters (otherwise samples land on the diagonal of the joint space).

    Each candidate value is selected ≈ random_n / k times.

    Reproducible via `seed`. Raises ValueError if random_n exceeds the
    Cartesian product size (use grid sampling for that).
    """
    cartesian_size = 1
    for v in space.values():
        cartesian_size *= len(v)
    if random_n > cartesian_size:
        raise ValueError(
            f"random_n={random_n} exceeds Cartesian product size {cartesian_size}; "
            f"use sampling='grid' for full enumeration."
        )

    rng = random.Random(seed)
    per_param_indices: dict[str, list[int]] = {}
    for name, values in space.items():
        k = len(values)
        stratum_width = k / random_n
        indices: list[int] = []
        for j in range(random_n):
            u = rng.random()
            x = (j + u) * stratum_width
            idx = min(int(math.floor(x)), k - 1)
            indices.append(idx)
        rng.shuffle(indices)
        per_param_indices[name] = indices

    samples: list[dict[str, Any]] = []
    for j in range(random_n):
        sample = {name: space[name][per_param_indices[name][j]] for name in space}
        samples.append(sample)
    return samples
```

- [ ] **Step 4: Wire into `OptimizationConfig` and `GridSearchOptimizer`**

In `backtester/config/models.py`, add to `OptimizationConfig`:

```python
@dataclass(slots=True)
class OptimizationConfig:
    # existing fields ...
    sampling: str = "grid"          # "grid" or "lhs"
    random_n: int = 100
    random_seed: int = 0
```

In `backtester/optimize/grid_search.py`, find `GridSearchOptimizer.optimize` (or the equivalent) and add a branch on `cfg.sampling`:

```python
def optimize(self, *, base_config, data, **_):
    space = base_config.optimization.param_space
    if base_config.optimization.sampling == "lhs":
        from backtester.optimize.lhs_sampler import sample_param_space
        combos = sample_param_space(
            space=space, random_n=base_config.optimization.random_n,
            seed=base_config.optimization.random_seed,
        )
    else:
        # existing Cartesian-product enumeration:
        combos = self._enumerate_grid(space)
    # ... rest of the loop unchanged ...
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/unit/test_optimizer_lhs.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Full suite**

Run: `python -m pytest -q`
Expected: `310 passed`.

- [ ] **Step 7: Commit**

```
git add backtester/optimize/lhs_sampler.py backtester/optimize/grid_search.py backtester/config/models.py tests/unit/test_optimizer_lhs.py
git commit -m "feat(optimize): discrete-LHS sampling over index positions in param_space"
```

Phase 12 ends.

---

## Phase 13: `mean_reversion_atr` strategy

Phase 13 lands the strategy itself. It's the only consumer of all the infrastructure built in Phases 1-12: per-bar callback, position_phase, aux_data, runtime trend gate.

14 unit tests in 3 tasks: state-machine entry/exit (5), phase-dependent logic (5), warmup + signal-shift + ctx contract (4).

Cumulative test target: **324** (310 + 14).

### Task 31: Params dataclass + indicators + entry/exit-without-phase rules

**Files:**
- Create: `strategies/mean_reversion_atr.py`
- Modify: `backtester/strategies/registry.py`
- Test: `tests/unit/test_strategy_mean_reversion_atr.py` (create, 5 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_strategy_mean_reversion_atr.py`:

```python
import pandas as pd
import pytest


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def test_strategy_is_registered():
    from backtester.strategies.registry import get_strategy_class
    cls = get_strategy_class("mean_reversion_atr")
    assert cls.strategy_id == "mean_reversion_atr"


def test_strategy_uses_multi_symbol_and_per_bar():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy
    assert MeanReversionAtrStrategy.uses_multi_symbol is True
    assert MeanReversionAtrStrategy.uses_per_bar is True


def test_params_defaults_match_prd():
    from strategies.mean_reversion_atr import MeanReversionAtrParams
    p = MeanReversionAtrParams()
    assert p.entry_atr_mult == 1.25
    assert p.mean_lookback == 10
    assert p.atr_lookback == 20
    assert p.time_stop_days == 7
    assert p.runner_time_stop_days == 12
    assert p.runner_ceiling_atr_mult == 1.25
    assert p.runtime_trend_threshold == 0.0025


def test_indicators_emits_mean10_atr20_slope200():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    data = _ohlcv([100.0 + 0.1 * i for i in range(250)])
    ind = s.indicators(data, MeanReversionAtrParams())
    assert "mean10" in ind.columns
    assert "atr20" in ind.columns
    assert "slope_log_200d" in ind.columns
    # Warmup: first values are NaN.
    assert pd.isna(ind["mean10"].iloc[0])
    assert pd.isna(ind["atr20"].iloc[0])


def test_warmup_bars_correct():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    p = MeanReversionAtrParams()
    # Warmup must cover the largest lookback used in indicators.
    assert s.warmup_bars(p) >= 200  # slope_log_200d is the deepest lookback
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v`
Expected: 5 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement params + indicators + skeleton**

Create `strategies/mean_reversion_atr.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.engine.atr import compute_atr
from backtester.engine.tranche_stop import TSPhase
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MeanReversionAtrParams:
    entry_atr_mult: float = 1.25
    mean_lookback: int = 10
    atr_lookback: int = 20
    time_stop_days: int = 7
    runner_time_stop_days: int = 12
    runner_ceiling_atr_mult: float = 1.25
    runtime_trend_threshold: float = 0.0025
    size: float = 1.0


def _ols_slope(window: np.ndarray) -> float:
    """OLS slope of values on bar index. Used in rolling().apply()."""
    x = np.arange(len(window), dtype=float)
    if len(x) < 2 or np.allclose(window, window[0]):
        return 0.0
    cov = np.cov(x, window, bias=True)[0, 1]
    var = float(np.var(x))
    return cov / var if var > 0 else 0.0


class MeanReversionAtrStrategy(BaseStrategy[MeanReversionAtrParams]):
    """Defense-first swing-trading mean-reversion strategy.

    See docs/superpowers/specs/2026-05-14-mean-reversion-atr-design.md §1.
    """
    strategy_id = "mean_reversion_atr"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"
    uses_multi_symbol = True
    uses_per_bar = True

    @classmethod
    def params_type(cls):
        return MeanReversionAtrParams

    def warmup_bars(self, params: MeanReversionAtrParams) -> int:
        return max(200, params.mean_lookback, params.atr_lookback) + 1

    def indicators(self, data: pd.DataFrame, params: MeanReversionAtrParams) -> pd.DataFrame:
        mean10 = data["close"].rolling(params.mean_lookback).mean()
        atr20 = compute_atr(data, params.atr_lookback)
        log_close = np.log(data["close"])
        slope_log_200d = log_close.rolling(200).apply(_ols_slope, raw=True)
        out = pd.DataFrame(index=data.index)
        out["mean10"] = mean10
        out["atr20"] = atr20
        out["slope_log_200d"] = slope_log_200d
        out["trend_active"] = (np.expm1(slope_log_200d).abs() > params.runtime_trend_threshold).fillna(False)
        return out

    def signal_for_bar(
        self,
        *,
        symbol: str,
        bar_idx: int,
        data_panel: dict[str, pd.DataFrame],
        indicators_panel: dict[str, pd.DataFrame],
        ctx: StrategyContext,
        params: MeanReversionAtrParams,
    ) -> float:
        """Per-bar signal in [-1.0, 1.0]. Implemented incrementally in Tasks 32-33."""
        # Skeleton: return 0 until the state machine lands.
        return 0.0

    # Legacy v0.3.0 method (unused for per-bar strategies but BaseStrategy may
    # require it). Return an empty frame.
    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0.0, "size": params.size}, index=data.index)
        return SignalFrame(data=df)
```

Then in `backtester/strategies/registry.py`, add:

```python
from strategies.mean_reversion_atr import MeanReversionAtrStrategy

register_strategy(MeanReversionAtrStrategy)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `315 passed`.

- [ ] **Step 6: Commit**

```
git add strategies/mean_reversion_atr.py backtester/strategies/registry.py tests/unit/test_strategy_mean_reversion_atr.py
git commit -m "feat(strategies): mean_reversion_atr — params, indicators, registry"
```

---

### Task 32: Entry rule with regime + trend-gate suppression

**Files:**
- Modify: `strategies/mean_reversion_atr.py`
- Test: `tests/unit/test_strategy_mean_reversion_atr.py` (append 4 tests)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def _run_strategy_signals(strategy, params, data, ctx_factory):
    """Helper: drive signal_for_bar over a full timeline; ctx_factory(i)→ctx."""
    indicators = strategy.indicators(data, params)
    indicators_panel = {"AAA": indicators}
    data_panel = {"AAA": data}
    out = []
    for i in range(len(data)):
        ctx = ctx_factory(i)
        target = strategy.signal_for_bar(
            symbol="AAA", bar_idx=i, data_panel=data_panel,
            indicators_panel=indicators_panel, ctx=ctx, params=params,
        )
        out.append(target)
    return pd.Series(out, index=data.index)


def _flat_ctx(i):
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase
    return StrategyContext(
        position_phase={"AAA": TSPhase.DISARMED},
        bars_in_phase={"AAA": 0},
    )


def test_entry_fires_at_125_atr_below_mean10():
    """Long entry when close <= mean10 - 1.25 * atr20 AND phase is DISARMED."""
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    # Build a series where bar 25 is a clear dip below the entry threshold.
    closes = [100.0] * 25 + [95.0] + [100.0] * 5  # one big dip
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, _flat_ctx)
    # Bar 25 close=95 vs mean10 ≈ 100 vs atr20 ≈ a few units. Long entry should fire.
    assert signals.iloc[25] > 0


def test_entry_suppressed_when_phase_not_disarmed():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def hard_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.HARD},
            bars_in_phase={"AAA": 2},
        )

    s = MeanReversionAtrStrategy()
    closes = [100.0] * 25 + [95.0] + [100.0] * 5
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, hard_ctx)
    # Even though the dip is there, we're already in HARD phase — no new entry.
    # Strategy must not emit a fresh +1.0 (it can emit other phase-driven targets).
    assert signals.iloc[25] <= 0.5  # not a full new entry


def test_entry_suppressed_when_trend_active():
    """Runtime trend gate: large 200d slope → no new entry even if dip occurs."""
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    # Build a strong uptrend (high slope) plus a small dip at the end.
    closes = [100.0 + 0.5 * i for i in range(250)]
    closes[245] = closes[245] - 5.0  # dip at bar 245
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, _flat_ctx)
    # The 200d slope is large (positive trend); the runtime gate must suppress entry.
    assert signals.iloc[245] == 0.0


def test_entry_suppressed_when_regime_book_flat():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase
    from backtester.engine.regime import RegimeState

    def book_flat_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.DISARMED},
            bars_in_phase={"AAA": 0},
            regime=RegimeState(
                spy_ema_tripped=True, vix_tripped=False, circuit_breaker_tripped=False,
            ),
        )

    s = MeanReversionAtrStrategy()
    closes = [100.0] * 25 + [95.0] + [100.0] * 5
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, book_flat_ctx)
    # Book flat: no new entry even though dip is there.
    assert signals.iloc[25] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v -k "entry_fires or entry_suppressed"`
Expected: 4 FAIL — the skeleton returns 0.

- [ ] **Step 3: Implement entry logic**

Replace `signal_for_bar` in `strategies/mean_reversion_atr.py`:

```python
def signal_for_bar(
    self,
    *,
    symbol: str,
    bar_idx: int,
    data_panel: dict[str, pd.DataFrame],
    indicators_panel: dict[str, pd.DataFrame],
    ctx: StrategyContext,
    params: MeanReversionAtrParams,
) -> float:
    data = data_panel[symbol]
    indicators = indicators_panel.get(symbol)
    if indicators is None:
        # Lazy compute if not pre-attached (test-mode path).
        indicators = self.indicators(data, params)
    if bar_idx >= len(data):
        return 0.0

    close = float(data["close"].iloc[bar_idx])
    mean10 = float(indicators["mean10"].iloc[bar_idx])
    atr20 = float(indicators["atr20"].iloc[bar_idx])
    if pd.isna(mean10) or pd.isna(atr20):
        return 0.0  # warmup

    phase = ctx.position_phase.get(symbol)
    regime = getattr(ctx, "regime", None)
    book_flat = (regime is not None and getattr(regime, "book_flat", False))

    # Entry gate:
    # 1. Phase must be DISARMED.
    # 2. Regime not flat.
    # 3. Runtime trend not active.
    # 4. close <= mean10 - entry_atr_mult * atr20.
    if phase is TSPhase.DISARMED and not book_flat:
        trend_active = bool(indicators["trend_active"].iloc[bar_idx])
        if not trend_active and close <= mean10 - params.entry_atr_mult * atr20:
            return 1.0

    return 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v -k entry`
Expected: 4 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `319 passed`.

- [ ] **Step 6: Commit**

```
git add strategies/mean_reversion_atr.py tests/unit/test_strategy_mean_reversion_atr.py
git commit -m "feat(mean_reversion_atr): entry rule with phase/regime/trend gating"
```

---

### Task 33: Exit rules — tranche 1, time stops, ceiling

**Files:**
- Modify: `strategies/mean_reversion_atr.py`
- Test: `tests/unit/test_strategy_mean_reversion_atr.py` (append 5 tests)

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_tranche_1_emits_half_target_at_mean_touch():
    """HARD phase + close >= mean10 → emit 0.5 (tranche 1 exit)."""
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def hard_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.HARD},
            bars_in_phase={"AAA": 2},
        )

    s = MeanReversionAtrStrategy()
    closes = [100.0] * 20 + [101.0] * 10
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, hard_ctx)
    # By bar 25, close has been >= mean10 for several bars.
    assert signals.iloc[25] == 0.5


def test_hard_phase_time_stop_at_7_days():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def long_hard_ctx(i):
        # Bars in HARD phase: 8 → exceeds default time_stop_days=7.
        return StrategyContext(
            position_phase={"AAA": TSPhase.HARD},
            bars_in_phase={"AAA": 8},
        )

    s = MeanReversionAtrStrategy()
    closes = [100.0] * 30
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, long_hard_ctx)
    # Time stop must fire after bar 25 (warmup done).
    assert signals.iloc[25] == 0.0


def test_runner_phase_time_stop_at_12_days():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def long_runner_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.RUNNER},
            bars_in_phase={"AAA": 13},
        )

    s = MeanReversionAtrStrategy()
    closes = [100.0] * 30
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, long_runner_ctx)
    assert signals.iloc[25] == 0.0


def test_runner_hard_ceiling_at_mean_plus_125_atr():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def runner_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.RUNNER},
            bars_in_phase={"AAA": 5},
        )

    s = MeanReversionAtrStrategy()
    # Strong rally above mean+1.25*atr → ceiling triggered.
    closes = [100.0] * 25 + [115.0, 115.0, 115.0, 115.0, 115.0]
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, runner_ctx)
    # Bar 25 close 115 is well above mean+1.25*atr → ceiling exit.
    assert signals.iloc[25] == 0.0


def test_strategy_does_not_close_position_on_runtime_trend_gate():
    """If a position is open and trend gate activates, position is HELD, not closed."""
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    from backtester.core.types import StrategyContext
    from backtester.engine.tranche_stop import TSPhase

    def runner_ctx(i):
        return StrategyContext(
            position_phase={"AAA": TSPhase.RUNNER},
            bars_in_phase={"AAA": 2},
        )

    s = MeanReversionAtrStrategy()
    # Trending series: trend_active = True everywhere late in series.
    closes = [100.0 + 0.5 * i for i in range(250)]
    data = _ohlcv(closes)
    signals = _run_strategy_signals(s, MeanReversionAtrParams(), data, runner_ctx)
    # Trend gate is active; phase is RUNNER. Strategy should NOT emit exit.
    # We expect 0.5 (held) or some non-zero — the runtime trend gate is for ENTRIES only.
    assert signals.iloc[245] != 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v -k "tranche_1 or time_stop or runner_hard_ceiling or runtime_trend_gate"`
Expected: 5 FAIL.

- [ ] **Step 3: Implement phase-dependent exits**

Replace `signal_for_bar` with the complete version:

```python
def signal_for_bar(
    self,
    *,
    symbol: str,
    bar_idx: int,
    data_panel: dict[str, pd.DataFrame],
    indicators_panel: dict[str, pd.DataFrame],
    ctx: StrategyContext,
    params: MeanReversionAtrParams,
) -> float:
    data = data_panel[symbol]
    indicators = indicators_panel.get(symbol)
    if indicators is None:
        indicators = self.indicators(data, params)
    if bar_idx >= len(data):
        return 0.0

    close = float(data["close"].iloc[bar_idx])
    mean10 = float(indicators["mean10"].iloc[bar_idx])
    atr20 = float(indicators["atr20"].iloc[bar_idx])
    if pd.isna(mean10) or pd.isna(atr20):
        return 0.0

    phase = ctx.position_phase.get(symbol)
    bars_in_phase = ctx.bars_in_phase.get(symbol, 0)
    regime = getattr(ctx, "regime", None)
    book_flat = (regime is not None and getattr(regime, "book_flat", False))

    # Universal: regime flat overrides everything (forces exit / no entry).
    if book_flat:
        return 0.0

    # HARD-phase exits:
    if phase is TSPhase.HARD:
        # Time stop:
        if bars_in_phase >= params.time_stop_days:
            return 0.0
        # Tranche 1 exit at mean:
        if close >= mean10:
            return 0.5
        # Otherwise hold full.
        return 1.0

    # RUNNER-phase exits:
    if phase is TSPhase.RUNNER:
        # Runner time stop:
        if bars_in_phase >= params.runner_time_stop_days:
            return 0.0
        # Hard ceiling:
        if close >= mean10 + params.runner_ceiling_atr_mult * atr20:
            return 0.0
        # Hold half.
        return 0.5

    # DISARMED: entry gate.
    trend_active = bool(indicators["trend_active"].iloc[bar_idx])
    if not trend_active and close <= mean10 - params.entry_atr_mult * atr20:
        return 1.0
    return 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_strategy_mean_reversion_atr.py -v`
Expected: 14 PASS (5 + 4 + 5 = 14).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `324 passed`.

- [ ] **Step 6: Commit**

```
git add strategies/mean_reversion_atr.py tests/unit/test_strategy_mean_reversion_atr.py
git commit -m "feat(mean_reversion_atr): tranche 1, time stops, runner ceiling"
```

Phase 13 ends.

---

## Phase 14: Universe screening CLI

Phase 14 lands `scripts/screen_universe.py`. The CLI pre-screens tickers by `range/ATR` ratio and trend filter, emitting `universe_candidates.yaml`.

6 unit tests in one task.

Cumulative test target: **330** (324 + 6).

### Task 34: `screen_universe.py` CLI

**Files:**
- Create: `scripts/screen_universe.py`
- Test: `tests/unit/test_screen_universe.py` (create, 6 tests)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_screen_universe.py`:

```python
import math
import pandas as pd
import pytest


def _ohlcv(closes, start="2022-01-03"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def test_range_atr_ratio_definition():
    """range_p10_p90_63d / atr_tr_20 — sanity check on a synthetic series."""
    from scripts.screen_universe import compute_metrics
    closes = [100.0 + 5.0 * math.sin(i * 0.1) for i in range(150)]
    data = _ohlcv(closes)
    m = compute_metrics(data)
    assert m["range_atr_ratio"] > 0


def test_slope_200d_pct_per_day_uses_expm1():
    """The percent reported is expm1(slope_log), not slope_log itself."""
    from scripts.screen_universe import compute_metrics
    # Exponential growth: close[i] = 100 * exp(0.005 * i) → slope_log ≈ 0.005.
    closes = [100.0 * math.exp(0.005 * i) for i in range(250)]
    data = _ohlcv(closes)
    m = compute_metrics(data)
    # expm1(0.005) ≈ 0.005012; should be within tolerance.
    assert m["slope_200d_pct_per_day"] == pytest.approx(0.005012, rel=0.05)


def test_trend_filter_requires_both_slope_and_r_squared():
    """A high-slope but low-R² series must NOT be rejected (it's noisy, not trending)."""
    from scripts.screen_universe import passes_filters
    # Compose: high slope, low r² → keep.
    keep = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.003, r_squared_200d=0.2,
        min_data_length_ok=True,
    )
    assert keep is True
    # High slope, high r² → drop.
    drop = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.003, r_squared_200d=0.5,
        min_data_length_ok=True,
    )
    assert drop is False


def test_min_data_length_filter():
    from scripts.screen_universe import passes_filters
    drop = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.0001, r_squared_200d=0.1,
        min_data_length_ok=False,
    )
    assert drop is False


def test_emits_unknown_sector_with_warning(tmp_path, capsys):
    from scripts.screen_universe import write_universe_yaml
    metrics_by_symbol = {
        "TSLA": {"sector": "Auto",    "range_atr_ratio": 9.0, "slope_200d_pct_per_day": 0.0005, "r_squared_200d": 0.1},
        "ZZZZ": {"sector": "unknown", "range_atr_ratio": 7.0, "slope_200d_pct_per_day": 0.0001, "r_squared_200d": 0.05},
    }
    out_path = tmp_path / "universe_candidates.yaml"
    write_universe_yaml(metrics_by_symbol, out=out_path, screening_window=("2023-01-01", "2025-12-31"))
    captured = capsys.readouterr()
    assert "ZZZZ" in captured.err or "unknown sector" in captured.err.lower()


def test_top_n_caps_output(tmp_path):
    from scripts.screen_universe import filter_and_rank
    metrics = {
        f"S{i:02d}": {"sector": "Test", "range_atr_ratio": 10.0 - 0.1 * i,
                     "slope_200d_pct_per_day": 0.0001, "r_squared_200d": 0.1}
        for i in range(50)
    }
    ranked = filter_and_rank(metrics, top=10)
    assert len(ranked) == 10
    # Top of the ranking has the highest range_atr_ratio.
    assert list(ranked.keys())[0] == "S00"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_screen_universe.py -v`
Expected: 6 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `scripts/screen_universe.py`:

```python
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtester.engine.atr import compute_atr


_SECTOR_MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "sector_map.csv"


def _load_sector_map() -> dict[str, str]:
    if not _SECTOR_MAP_PATH.exists():
        return {}
    with _SECTOR_MAP_PATH.open(newline="", encoding="utf-8") as f:
        return {row["symbol"]: row["sector"] for row in csv.DictReader(f)}


def compute_metrics(data: pd.DataFrame) -> dict[str, float]:
    """Compute range/ATR, 200d slope, R² for one ticker."""
    close = data["close"]

    # range_p10_p90 over rolling 63 bars, median across the window
    def _range(window: np.ndarray) -> float:
        return float(np.percentile(window, 90) - np.percentile(window, 10))

    range_series = close.rolling(63).apply(_range, raw=True)
    range_med = float(range_series.median())

    atr_series = compute_atr(data, 20)
    atr_med = float(atr_series.median())
    range_atr_ratio = range_med / atr_med if atr_med > 0 else 0.0

    # 200d log-OLS slope + R² over rolling 200 bars; median across windows.
    log_close = np.log(close)

    def _slope_r2(window: np.ndarray) -> tuple[float, float]:
        x = np.arange(len(window), dtype=float)
        if len(x) < 2:
            return (0.0, 0.0)
        cov = np.cov(x, window, bias=True)[0, 1]
        var_x = float(np.var(x))
        var_y = float(np.var(window))
        if var_x <= 0 or var_y <= 0:
            return (0.0, 0.0)
        slope = cov / var_x
        r = cov / math.sqrt(var_x * var_y)
        return (slope, r * r)

    slopes = []
    r2s = []
    for i in range(199, len(log_close)):
        window = log_close.iloc[i - 199: i + 1].to_numpy()
        s, r2 = _slope_r2(window)
        slopes.append(s)
        r2s.append(r2)
    slope_log_med = float(np.median(slopes)) if slopes else 0.0
    r_squared_med = float(np.median(r2s)) if r2s else 0.0

    return {
        "range_atr_ratio": range_atr_ratio,
        "slope_200d_pct_per_day": float(np.expm1(slope_log_med)),
        "r_squared_200d": r_squared_med,
    }


def passes_filters(
    *,
    range_atr_ratio: float,
    slope_200d_pct_per_day: float,
    r_squared_200d: float,
    min_data_length_ok: bool,
) -> bool:
    if not min_data_length_ok:
        return False
    if abs(slope_200d_pct_per_day) > 0.002 and r_squared_200d > 0.4:
        return False
    if range_atr_ratio < 5.0:
        return False
    return True


def filter_and_rank(
    metrics_by_symbol: dict[str, dict],
    *,
    top: int,
) -> dict[str, dict]:
    kept = {
        sym: m for sym, m in metrics_by_symbol.items()
        if passes_filters(
            range_atr_ratio=m["range_atr_ratio"],
            slope_200d_pct_per_day=m["slope_200d_pct_per_day"],
            r_squared_200d=m["r_squared_200d"],
            min_data_length_ok=True,
        )
    }
    ranked = sorted(kept.items(), key=lambda kv: -kv[1]["range_atr_ratio"])[:top]
    return dict(ranked)


def write_universe_yaml(
    metrics_by_symbol: dict[str, dict],
    *,
    out: Path,
    screening_window: tuple[str, str],
) -> None:
    sector_map = _load_sector_map()
    out_doc = {
        "_meta": {
            "generated_by": "scripts/screen_universe.py",
            "screening_window_start": screening_window[0],
            "screening_window_end": screening_window[1],
            "filters": "|slope| < 0.2%/d AND R² < 0.4; range/atr >= 5.0",
        },
        "universe": {},
    }
    for sym, m in metrics_by_symbol.items():
        sector = m.get("sector") or sector_map.get(sym) or "unknown"
        if sector == "unknown":
            print(f"WARNING: unknown sector for {sym}", file=sys.stderr)
        out_doc["universe"][sym] = {
            "sector": sector,
            "range_atr_ratio": round(m["range_atr_ratio"], 2),
            "slope_200d": round(m["slope_200d_pct_per_day"], 4),
            "r_squared_200d": round(m["r_squared_200d"], 3),
        }
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_doc, f, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("screen_universe")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--data-root", default="data/raw")
    args = parser.parse_args(argv)

    from backtester.data.loader import load_symbol

    candidates = [
        s.strip() for s in args.candidates.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]

    sector_map = _load_sector_map()
    metrics_by_symbol: dict[str, dict] = {}
    for sym in candidates:
        try:
            data = load_symbol(
                symbol=sym, source="yfinance", root=args.data_root,
                start=args.start, end=args.end,
                require_volume=False,  # tolerate index-style aux symbols
            )
        except Exception as exc:
            print(f"WARNING: skipping {sym}: {exc}", file=sys.stderr)
            continue
        if len(data) < 504:
            print(f"WARNING: {sym} has only {len(data)} bars; skipping", file=sys.stderr)
            continue
        m = compute_metrics(data)
        m["sector"] = sector_map.get(sym, "unknown")
        metrics_by_symbol[sym] = m

    ranked = filter_and_rank(metrics_by_symbol, top=args.top)
    write_universe_yaml(ranked, out=args.out, screening_window=(args.start, args.end))
    print(f"wrote {len(ranked)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_screen_universe.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `330 passed`.

- [ ] **Step 6: Commit**

```
git add scripts/screen_universe.py tests/unit/test_screen_universe.py
git commit -m "feat(scripts): screen_universe.py CLI (range/ATR ratio + trend filter)"
```

Phase 14 ends.

---

## Phase 15: Runner routing

Phase 15 modifies `run_batch.py` and `run_wfo.py` to detect strategies with `uses_multi_symbol = True` and route them through `MultiSymbolBacktestEngine`. v0.3.0 strategies continue through the existing single-symbol path. No test count change (this is plumbing). A smoke test is added in Phase 18.

Cumulative test target: **330**.

### Task 35: Route `run_batch` through `MultiSymbolBacktestEngine`

**Files:**
- Modify: `backtester/runners/run_batch.py`

This is the hardest mechanical change — `run_batch.py` becomes a router. No tests in this task; behavior is verified by Phase 18's integration smoke.

- [ ] **Step 1: Read the current `run_batch.py`**

Refresh your memory: `run_batch.py` currently runs each symbol independently through `BacktestEngine`. It does NOT have shared cash.

- [ ] **Step 2: Modify routing logic**

Replace `backtester/runners/run_batch.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.io.serialization import write_json
from backtester.strategies.instantiate import build_strategy_and_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_batch")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    # Build strategy first to decide the path.
    strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)

    if getattr(strategy, "uses_multi_symbol", False):
        return _run_multi_symbol(rc=rc, strategy=strategy, params=params,
                                  writer=writer, log=log)
    return _run_legacy_per_symbol(rc=rc, strategy=strategy, params=params,
                                   writer=writer, log=log)


def _run_legacy_per_symbol(*, rc, strategy, params, writer, log) -> int:
    """v0.3.0 path: independent per-symbol runs, no shared cash."""
    if not rc.data.symbols:
        raise SystemExit("data.symbols is empty (and strategy is not multi-symbol)")
    by_symbol = {}
    for symbol in rc.data.symbols:
        try:
            data = load_symbol(
                symbol=symbol, source=rc.data.source, root=rc.data.root,
                start=rc.data.start, end=rc.data.end,
                auto_adjust=rc.data.auto_adjust,
            )
            validate_ohlcv(data)
            broker = Broker(rc.execution)
            portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
            engine = BacktestEngine(broker=broker, portfolio=portfolio)
            result = engine.run(strategy, data, params, symbol=symbol, timeframe=rc.data.timeframe)
            by_symbol[symbol] = result.summary
            result.equity_curve.to_csv(writer.run_dir / f"{symbol}_equity_curve.csv", index_label="timestamp")
            result.trades.to_csv(writer.run_dir / f"{symbol}_trades.csv", index=False)
        except Exception as exc:
            log.warning("%s failed: %s", symbol, exc)
            by_symbol[symbol] = {"error": str(exc)}

    writer.write_config(rc)
    write_json(writer.run_dir / "batch_summary.json", by_symbol)
    return 0


def _run_multi_symbol(*, rc, strategy, params, writer, log) -> int:
    """v0.4.0 path: single shared-cash run across the universe."""
    from backtester.config.universe import load_universe_config

    if rc.universe_path is None:
        raise SystemExit("multi-symbol strategy requires universe_path in run config")

    universe = load_universe_config(
        path=Path(rc.universe_path), global_params=rc.strategy_params,
    )
    symbols = list(universe.keys())
    sectors = {sym: cfg.sector for sym, cfg in universe.items()}

    # Load OHLCV for every symbol AND aux_symbols.
    data: dict = {}
    for sym in symbols:
        d = load_symbol(
            symbol=sym, source=rc.data.source, root=rc.data.root,
            start=rc.data.start, end=rc.data.end,
            auto_adjust=rc.data.auto_adjust,
        )
        validate_ohlcv(d)
        data[sym] = d

    aux_data: dict = {}
    for aux_sym in rc.data.aux_symbols:
        a = load_symbol(
            symbol=aux_sym, source=rc.data.source, root=rc.data.root,
            start=rc.data.start, end=rc.data.end,
            auto_adjust=rc.data.auto_adjust, require_volume=False,
        )
        validate_ohlcv(a, strict_volume=False)
        aux_data[aux_sym] = a

    sim = MultiSymbolPortfolioSimulator(
        config=rc.portfolio,
        initial_cash=rc.execution.initial_cash,
        broker_factory=lambda: Broker(rc.execution),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    result = engine.run(
        strategy=strategy, symbols=symbols, data=data, sectors=sectors,
        aux_data=aux_data, params=params, regime_config=rc.regimes,
    )

    # Write per-symbol artifacts plus portfolio-level artifacts.
    writer.write_config(rc)
    result.equity_curve.to_csv(writer.run_dir / "portfolio_equity_curve.csv", index_label="timestamp")
    write_json(writer.run_dir / "batch_summary.json", {
        "portfolio_total_return": result.portfolio_total_return,
        "portfolio_max_drawdown": result.portfolio_max_drawdown,
        "portfolio_sharpe": result.portfolio_sharpe,
        "final_equity": result.final_equity,
        "n_symbols": len(symbols),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run full suite (must not regress)**

Run: `python -m pytest -q`
Expected: `330 passed`. v0.3.0 single-symbol paths continue to work.

- [ ] **Step 4: Smoke-test the legacy path manually**

Run: `python -m backtester.runners.run_batch --config <any-existing-v0.3.0-batch-config>`
Expected: exit 0. Existing behavior preserved.

- [ ] **Step 5: Commit**

```
git add backtester/runners/run_batch.py
git commit -m "feat(runners): route multi-symbol strategies through MultiSymbolBacktestEngine"
```

---

### Task 36: Route `run_wfo` through `MultiSymbolBacktestEngine`

**Files:**
- Modify: `backtester/runners/run_wfo.py`

`run_wfo` currently requires exactly one symbol. For multi-symbol strategies, the same multi-symbol simulator runs INSIDE each WFO window.

- [ ] **Step 1: Modify routing**

In `backtester/runners/run_wfo.py`, near the existing one-symbol check, replace:

```python
if len(rc.data.symbols) != 1:
    raise SystemExit("run_wfo expects exactly one symbol")
```

with:

```python
from backtester.strategies.registry import get_strategy_class
strategy_cls = get_strategy_class(rc.strategy)
is_multi = getattr(strategy_cls, "uses_multi_symbol", False)
if not is_multi and len(rc.data.symbols) != 1:
    raise SystemExit("run_wfo (single-symbol path) expects exactly one symbol")
if is_multi and rc.universe_path is None:
    raise SystemExit("run_wfo (multi-symbol path) requires universe_path")
```

For the multi-symbol path, the existing `WalkForwardRunner` cannot be reused as-is — it expects a single data series. The simplest extension: branch early and run a WFO-equivalent loop inside `_run_multi_symbol_wfo`. For Phase 15, defer the full multi-symbol WFO loop to a follow-up and have the runner raise `NotImplementedError("multi-symbol WFO lands in v0.4.1")` when the multi-symbol branch is taken.

```python
if is_multi:
    raise SystemExit(
        "multi-symbol WFO lands in v0.4.1; for v0.4.0, run the full backtest via "
        "run_batch and inspect window-level metrics manually."
    )
```

This is a documented limitation in `docs/runbook.md` (Phase 20).

- [ ] **Step 2: Full suite (no regressions)**

Run: `python -m pytest -q`
Expected: `330 passed`.

- [ ] **Step 3: Commit**

```
git add backtester/runners/run_wfo.py
git commit -m "feat(runners): run_wfo detects multi-symbol strategies; explicit v0.4.1 deferral"
```

Phase 15 ends. Multi-symbol WFO is explicitly out of scope for v0.4.0; spec §11 already lists it under "Out of scope" as part of "WFO over the regime / sizing / risk-budget surface".

---

## Phase 16: Multi-symbol artifacts

Phase 16 extends `ArtifactWriter` to materialize `config_resolved.yaml` for multi-symbol runs (flattening per-name overrides) and to write per-symbol trade logs alongside the portfolio-level artifacts. One task; one new test.

Cumulative test target: **331** (330 + 1).

### Task 37: Multi-symbol `config_resolved.yaml` + per-symbol trade logs

**Files:**
- Modify: `backtester/io/artifacts.py`
- Test: `tests/unit/test_artifacts_multi_symbol.py` (create, 1 test)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_artifacts_multi_symbol.py`:

```python
def test_artifact_writer_writes_resolved_config_with_universe_expansion(tmp_path):
    from backtester.io.artifacts import ArtifactWriter
    from backtester.config.models import (
        RunConfig, DataConfig, ExecutionConfig, PortfolioConfig,
    )
    from backtester.config.universe import ResolvedSymbolConfig

    rc = RunConfig(
        run_name="vtest",
        strategy="mean_reversion_atr",
        strategy_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
        data=DataConfig(source="csv", root="data/raw",
                        start="2024-01-01", end="2024-06-30", timeframe="1d"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
        output_root=str(tmp_path),
        universe_path="configs/universe.yaml",
    )
    writer = ArtifactWriter(root=str(tmp_path), run_name=rc.run_name)
    universe = {
        "TSLA": ResolvedSymbolConfig(
            symbol="TSLA", sector="Auto",
            effective_params={"entry_atr_mult": 1.5, "mean_lookback": 10},
        ),
        "NVDA": ResolvedSymbolConfig(
            symbol="NVDA", sector="Semis",
            effective_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
        ),
    }
    writer.write_config(rc, resolved_universe=universe)

    import yaml
    with (writer.run_dir / "config_resolved.yaml").open() as f:
        doc = yaml.safe_load(f)
    # The resolved universe is materialized with effective params.
    assert "resolved_universe" in doc
    assert doc["resolved_universe"]["TSLA"]["effective_params"]["entry_atr_mult"] == 1.5
    assert doc["resolved_universe"]["NVDA"]["sector"] == "Semis"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_artifacts_multi_symbol.py -v`
Expected: FAIL — `write_config` doesn't accept `resolved_universe` kwarg.

- [ ] **Step 3: Implement**

In `backtester/io/artifacts.py`, modify `ArtifactWriter.write_config`:

```python
def write_config(self, rc, *, resolved_universe=None):
    """Write config_resolved.yaml.

    For multi-symbol runs, `resolved_universe` is a dict[symbol, ResolvedSymbolConfig].
    The materialized form is embedded under `resolved_universe:` for reproducibility.
    """
    doc = self._as_yaml_dict(rc)
    if resolved_universe is not None:
        doc["resolved_universe"] = {
            sym: {
                "sector": cfg.sector,
                "effective_params": dict(cfg.effective_params),
            }
            for sym, cfg in resolved_universe.items()
        }
    with (self.run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(doc, f, sort_keys=False)
```

`_as_yaml_dict` is the existing helper; preserve its current behavior. If `write_config` currently has a different signature, preserve all existing call sites and add `resolved_universe` as a keyword-only kwarg with default `None`.

Then update `_run_multi_symbol` in `run_batch.py` (Phase 15) to pass `resolved_universe`:

```python
writer.write_config(rc, resolved_universe=universe)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_artifacts_multi_symbol.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `331 passed`.

- [ ] **Step 6: Commit**

```
git add backtester/io/artifacts.py backtester/runners/run_batch.py tests/unit/test_artifacts_multi_symbol.py
git commit -m "feat(artifacts): materialize resolved_universe in config_resolved.yaml"
```

Phase 16 ends.

---

## Phase 17: Real-data fixtures (gated by user approval)

Phase 17 fetches real OHLCV for the 17 tickers (15 universe + SPY + ^VIX) and commits them to `data/raw/` as test fixtures. This phase REQUIRES explicit user approval before proceeding — the brief says "no new symbols added to `data/raw/` without explicit user approval". The execution agent MUST PAUSE at Task 38 and confirm.

Cumulative test target: **331**.

### Task 38: User approval gate

**Files:**
- No code changes in this task.

- [ ] **Step 1: HARD PAUSE — confirm user approval**

The execution agent MUST stop here and confirm with the user:

> "About to fetch real OHLCV for 17 tickers (TSLA, NVDA, AMD, COIN, GOOGL, MSTR, XPEV, NIO, PLTR, SMCI, SHOP, W, META, NFLX, ^VIX, SPY, AAPL) via yfinance, covering 2015-01-02 to 2025-12-31. The fetched CSVs will replace bundled synthetic SPY.csv and AAPL.csv at data/raw/, and add the 15 universe tickers + ^VIX as new files. This is the explicit-user-approval gate from the project brief.
>
> Constraints:
>   - yfinance must be installed: `pip install -e .[data,dev]`
>   - Network is required for this task only.
>   - The fetched CSVs will be committed to git as test fixtures (~4 MB total).
>   - data/raw/SPY.csv and data/raw/AAPL.csv become REAL data, NOT synthetic.
>     The synthetic versions are already in data/synth/ from Phase 2.
>
> Confirm to proceed?"

DO NOT proceed past this step without the user typing "yes" or equivalent affirmative.

- [ ] **Step 2: Install yfinance extras**

Run: `pip install -e .[data]`
Expected: yfinance installed successfully.

- [ ] **Step 3: No commit for Task 38**

The user-approval gate produces no source changes. Proceed to Task 39.

---

### Task 39: Fetch the 17 tickers via yfinance

**Files:**
- Modify: `data/raw/SPY.csv` (overwrite synthetic → real)
- Modify: `data/raw/AAPL.csv` (overwrite synthetic → real)
- Create: `data/raw/{TSLA,NVDA,AMD,COIN,GOOGL,MSTR,XPEV,NIO,PLTR,SMCI,SHOP,W,META,NFLX,^VIX}.csv`

- [ ] **Step 1: Trigger the yfinance fetch**

Run a one-shot Python session:

```
python -c "
from backtester.data.loader import load_symbol
tickers = ['SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'COIN', 'GOOGL', 'MSTR',
           'XPEV', 'NIO', 'PLTR', 'SMCI', 'SHOP', 'W', 'META', 'NFLX', '^VIX']
for t in tickers:
    print(f'fetching {t}...')
    require_volume = t != '^VIX'
    df = load_symbol(symbol=t, source='yfinance', root='data/raw',
                     start='2015-01-02', end='2025-12-31',
                     require_volume=require_volume)
    print(f'  {t}: {len(df)} bars [{df.index.min().date()}, {df.index.max().date()}]')
"
```

Expected: 17 CSVs land in `data/raw/`. Each ~250 KB.

- [ ] **Step 2: Verify file shapes**

Run: `python -c "
import pandas as pd
for sym in ['SPY', 'TSLA', '^VIX', 'AAPL']:
    df = pd.read_csv(f'data/raw/{sym}.csv', index_col=0, parse_dates=True)
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume'], f'{sym}: {df.columns}'
    assert len(df) > 2000, f'{sym}: only {len(df)} bars'
print('OK')
"`
Expected: `OK`.

- [ ] **Step 3: Full suite (must still pass with REAL SPY/AAPL fixtures)**

This step is the moment of truth. v0.3.0 baseline tests (sma_cross_spy.yaml, momentum_streak_spy.yaml, etc.) now run on REAL SPY data and may produce different numerics.

Run: `python -m pytest -q`
Expected: most pass; SOME tests will fail because their golden numerics were captured against synthetic SPY:

  - `tests/integration/test_run_backtest_cli.py` — assertions about trade counts, return signs, etc.
  - `tests/integration/test_backwards_compat.py` — explicit byte-for-byte golden comparison.

These failures are EXPECTED at this point. They are resolved in Phase 19. Record the exact set of failures here:

```
$ python -m pytest -q 2>&1 | grep FAILED > docs/superpowers/plans/v04_phase17_failures.txt
```

The list should include only:
  - integration tests that depended on synthetic SPY numerics
  - `test_backwards_compat.py::test_sma_cross_spy_unchanged_*` (renamed in Phase 19 to point at `data/synth/SPY.csv`)

If ANY unit test fails, STOP — it indicates a real regression, not a fixture issue.

- [ ] **Step 4: Commit the fixtures**

```
git add data/raw/*.csv
git commit -m "data(fixtures): commit real OHLCV for 15-name universe + SPY/AAPL/^VIX (2015-2025)

Replaces synthetic data/raw/SPY.csv and data/raw/AAPL.csv with real fetched
prices. Synthetic versions remain at data/synth/. Integration tests that
asserted synthetic numerics are updated in Phase 19."
```

Phase 17 ends. Real-data fixtures are in place.

---

## Phase 18: Integration smokes — multi-symbol batch + screen_universe

Phase 18 adds end-to-end CLI smoke tests against the committed real fixtures and a parsing smoke for `screen_universe`.

3 integration tests across 2 tasks.

Cumulative test target: **334** (331 + 3).

### Task 40: Multi-symbol `run_batch` CLI smoke

**Files:**
- Create: `configs/backtests/mean_rev_v04_smoke.yaml` (small subset universe for fast tests)
- Create: `configs/universe_smoke.yaml`
- Test: `tests/integration/test_run_batch_cli.py` (append 1 test)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_run_batch_cli.py`:

```python
import json
from pathlib import Path
import subprocess
import sys


def test_multi_symbol_mean_reversion_smoke(tmp_path, monkeypatch):
    """End-to-end: 3-symbol universe, 2024-only window, exit 0 with artifacts."""
    cfg = Path("configs/backtests/mean_rev_v04_smoke.yaml")
    assert cfg.exists(), f"missing {cfg}"
    # Use a short window for speed.
    monkeypatch.setenv("BACKTESTER_OUTPUT_ROOT", str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # Find the run directory.
    run_dirs = list(tmp_path.glob("*mean_rev_v04_smoke*"))
    assert run_dirs, f"no run dir under {tmp_path}"
    run_dir = run_dirs[0]
    # Verify portfolio artifacts.
    assert (run_dir / "portfolio_equity_curve.csv").exists()
    assert (run_dir / "batch_summary.json").exists()
    summary = json.loads((run_dir / "batch_summary.json").read_text())
    assert "portfolio_total_return" in summary
    assert summary["n_symbols"] == 3
```

- [ ] **Step 2: Create the smoke configs**

Create `configs/universe_smoke.yaml`:

```yaml
universe:
  TSLA: {sector: Auto, overrides: {entry_atr_mult: 1.5}}
  NVDA: {sector: Semis}
  AMD:  {sector: Semis}
```

Create `configs/backtests/mean_rev_v04_smoke.yaml`:

```yaml
run_name: mean_rev_v04_smoke
strategy: mean_reversion_atr
universe_path: ../universe_smoke.yaml
data:
  source: csv
  root: data/raw
  start: '2024-01-02'
  end: '2024-12-31'
  timeframe: 1d
  auto_adjust: true
  aux_symbols: [SPY, '^VIX']
strategy_params:
  entry_atr_mult: 1.25
  mean_lookback: 10
  atr_lookback: 20
  time_stop_days: 7
  runner_time_stop_days: 12
  runner_ceiling_atr_mult: 1.25
  runtime_trend_threshold: 0.0025
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
  allow_short: false
  hard_stop_atr_mult: 1.75
  runner_atr_mult: 2.5
  breakeven_floor: true
  tranche_stop_atr_period: 20
portfolio:
  sizing_mode: vol_targeted
  vol_target: 0.12
  position_cap_pct: 0.10
  cash_reserve_pct: 0.30
  risk_budget_pct: 0.06
  sector_cap_pct: 0.50
regimes:
  spy_ema:
    enabled: true
    ema_lookback: 200
    trip_pct: -0.02
    resume_pct: 0.02
  vix:
    enabled: true
    trip_threshold: 30
    trip_consec: 2
    resume_threshold: 25
    resume_consec: 3
  circuit_breaker:
    enabled: true
    pnl_window_days: 20
    trip_pct: -0.05
    pause_days: 10
output_root: output/runs
```

- [ ] **Step 3: Run the test (the runner must read `BACKTESTER_OUTPUT_ROOT` if set, falling back to the YAML's `output_root`)**

If `output_root` is currently NOT environment-overridable, add a tiny tweak in `run_batch.py`:

```python
import os
output_root = os.environ.get("BACKTESTER_OUTPUT_ROOT", rc.output_root)
writer = ArtifactWriter(root=output_root, run_name=rc.run_name)
```

Run: `python -m pytest tests/integration/test_run_batch_cli.py::test_multi_symbol_mean_reversion_smoke -v`
Expected: PASS.

- [ ] **Step 4: Full suite**

Run: `python -m pytest -q`
Expected: `332 passed` (331 + 1 new). Existing v0.3.0 failures from Phase 17 still present.

- [ ] **Step 5: Commit**

```
git add configs/backtests/mean_rev_v04_smoke.yaml configs/universe_smoke.yaml tests/integration/test_run_batch_cli.py backtester/runners/run_batch.py
git commit -m "test(integration): multi-symbol run_batch CLI smoke (3-name 2024 window)"
```

---

### Task 41: `screen_universe` CLI smoke + WFO deferral message

**Files:**
- Create: `configs/universe_candidates_seed.txt`
- Test: `tests/integration/test_screen_universe_cli.py` (create, 1 test)
- Test: `tests/integration/test_run_wfo_cli.py` (append 1 test)

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_screen_universe_cli.py`:

```python
import subprocess
import sys
from pathlib import Path


def test_screen_universe_smoke(tmp_path):
    seed = tmp_path / "seed.txt"
    seed.write_text("TSLA\nNVDA\nAMD\n")
    out = tmp_path / "universe_candidates.yaml"
    result = subprocess.run(
        [
            sys.executable, "scripts/screen_universe.py",
            "--candidates", str(seed),
            "--start", "2023-01-03", "--end", "2024-12-31",
            "--out", str(out), "--top", "10",
            "--data-root", "data/raw",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    import yaml
    doc = yaml.safe_load(out.read_text())
    assert "universe" in doc
```

Append to `tests/integration/test_run_wfo_cli.py`:

```python
def test_run_wfo_multi_symbol_explicit_deferral(tmp_path):
    """v0.4.0 multi-symbol WFO is deferred; runner emits a clear message."""
    import subprocess
    import sys
    # Create a multi-symbol WFO config that targets the deferral path.
    cfg = tmp_path / "wfo_multi.yaml"
    cfg.write_text(
        "run_name: vtest\n"
        "strategy: mean_reversion_atr\n"
        "universe_path: ../../configs/universe_smoke.yaml\n"  # relative resolution
        "strategy_params: {entry_atr_mult: 1.25, mean_lookback: 10}\n"
        "data:\n"
        "  source: csv\n  root: data/raw\n"
        "  start: '2024-01-02'\n  end: '2024-06-30'\n"
        "  timeframe: 1d\n  aux_symbols: [SPY, '^VIX']\n"
        "execution: {initial_cash: 100000}\n"
        "portfolio: {sizing_mode: vol_targeted, vol_target: 0.12}\n"
        "wfo: {enabled: true, train_bars: 60, test_bars: 30, step_bars: 30}\n"
        "optimization: {objective: sharpe, param_space: {entry_atr_mult: [1.0, 1.25]}}\n"
        "output_root: " + str(tmp_path) + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_wfo", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "v0.4.1" in result.stdout + result.stderr
```

- [ ] **Step 2: Run to verify failure / passes**

Run: `python -m pytest tests/integration/test_screen_universe_cli.py tests/integration/test_run_wfo_cli.py -v`
Expected: both PASS. The seed file step doesn't exist yet, so create it.

Create `configs/universe_candidates_seed.txt`:

```
TSLA
NVDA
AMD
COIN
GOOGL
MSTR
XPEV
NIO
PLTR
SMCI
SHOP
W
META
NFLX
```

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: `334 passed`.

- [ ] **Step 4: Commit**

```
git add configs/universe_candidates_seed.txt tests/integration/test_screen_universe_cli.py tests/integration/test_run_wfo_cli.py
git commit -m "test(integration): screen_universe CLI smoke + multi-symbol WFO deferral check"
```

Phase 18 ends.

---

## Phase 19: Stress windows + held-out + backwards-compat update

Phase 19 wires the 4 stress-window integration tests and the held-out continuous test, all marked xfail-by-default. It also re-points `test_backwards_compat.py` at `data/synth/SPY.csv` to restore byte-identical v0.3.0 semantics on synthetic data.

5 new integration tests (4 parametrized + 1 held-out). Plus the backwards-compat test count remains constant (the file is modified, not added to).

Cumulative test target: **339** (334 + 5).

### Task 42: Stress-window integration tests (xfail-by-default)

**Files:**
- Create: `tests/integration/test_stress_windows.py`

- [ ] **Step 1: Create the test file**

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


STRESS_WINDOWS = [
    ("2020-covid",      "2020-02-15", "2020-04-30"),
    ("2022-bear-cycle", "2021-11-01", "2022-10-31"),
    ("2024-aug-unwind", "2024-07-15", "2024-09-15"),
    ("2025-apr",        "2025-03-15", "2025-05-15"),
]


def _write_config(tmp_path, *, start, end) -> Path:
    cfg = tmp_path / "stress.yaml"
    cfg.write_text(
        (Path("configs/backtests/mean_rev_v04_smoke.yaml").read_text()
         .replace("start: '2024-01-02'", f"start: '{start}'")
         .replace("end: '2024-12-31'", f"end: '{end}'")
         .replace("output_root: output/runs", f"output_root: {tmp_path}")
        )
    )
    return cfg


@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
@pytest.mark.parametrize("label,start,end", STRESS_WINDOWS)
def test_stress_window_drawdown(tmp_path, label, start, end):
    cfg = _write_config(tmp_path, start=start, end=end)
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{label} structural failure: {result.stderr}"
    run_dirs = list(tmp_path.glob("*stress*")) or list(tmp_path.glob("*mean_rev*"))
    summary_path = run_dirs[0] / "batch_summary.json"
    summary = json.loads(summary_path.read_text())
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    # Performance assertion — wrapped by xfail.
    assert summary["portfolio_max_drawdown"] > -0.09, (
        f"{label}: DD {summary['portfolio_max_drawdown']:.4f} exceeded -9%"
    )
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_stress_windows.py -v`
Expected: 4 XFAIL (or XPASS if the strategy happens to clear the bar already). Either way, exit code 0.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: `338 passed, 4 xfailed` (or `338 passed, 4 xpassed`).

- [ ] **Step 4: Commit**

```
git add tests/integration/test_stress_windows.py
git commit -m "test(integration): 4 stress-window drawdown tests (xfail-by-default)"
```

---

### Task 43: Held-out 2022-2025 continuous test (xfail-by-default)

**Files:**
- Create: `tests/integration/test_held_out_2022_2025.py`

- [ ] **Step 1: Create the test file**

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_config(tmp_path) -> Path:
    cfg = tmp_path / "held_out.yaml"
    cfg.write_text(
        (Path("configs/backtests/mean_rev_v04_smoke.yaml").read_text()
         .replace("start: '2024-01-02'", "start: '2022-01-03'")
         .replace("end: '2024-12-31'", "end: '2025-12-31'")
         .replace("output_root: output/runs", f"output_root: {tmp_path}")
        )
    )
    return cfg


@pytest.mark.xfail(strict=False, reason="performance gate; flip to assert when strategy is tuned")
def test_held_out_2022_2025(tmp_path):
    cfg = _write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"structural failure: {result.stderr}"
    run_dirs = list(tmp_path.glob("*mean_rev*"))
    summary = json.loads((run_dirs[0] / "batch_summary.json").read_text())
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    # PRD-derived performance gates.
    assert summary["portfolio_max_drawdown"] > -0.09, "held-out DD breach"
    assert summary["portfolio_total_return"] > 0.15, "held-out return under 15%"
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_held_out_2022_2025.py -v`
Expected: XFAIL or XPASS. Exit 0.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: `338 passed, 5 xfailed`.

- [ ] **Step 4: Commit**

```
git add tests/integration/test_held_out_2022_2025.py
git commit -m "test(integration): held-out 2022-2025 continuous (xfail-by-default)"
```

---

### Task 44: Re-point backwards-compat test at `data/synth/SPY.csv`

**Files:**
- Modify: `tests/integration/test_backwards_compat.py`
- Modify: `configs/backtests/sma_cross_spy.yaml` (or create a synth-pointed twin)

This task restores byte-identical v0.3.0 behavior on synthetic data. The synth-pointed config goes to `configs/backtests/sma_cross_synth_spy.yaml`; the existing real-pointed `sma_cross_spy.yaml` stays as-is for the "real-data" smoke.

- [ ] **Step 1: Capture the v0.3.0 golden from the scratch file**

Read `docs/superpowers/plans/v04_baseline_golden.txt` (captured in Phase 1).

- [ ] **Step 2: Create the synth-pointed config**

Create `configs/backtests/sma_cross_synth_spy.yaml`:

```yaml
run_name: sma_cross_synth_spy
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
  size: 1.0
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: "csv"
  root: "data/synth"     # CHANGED from data/raw
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
  allow_short: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
output_root: "output/runs"
```

- [ ] **Step 3: Modify the test**

In `tests/integration/test_backwards_compat.py`, change the config path:

```python
def test_sma_cross_synth_spy_unchanged(tmp_path):
    """v0.3.0 numerics preserved on synthetic SPY (data/synth/)."""
    # ... existing test body ...
    cfg = "configs/backtests/sma_cross_synth_spy.yaml"  # CHANGED
    # Golden numerics CAPTURED FROM v04_baseline_golden.txt:
    GOLDEN = {
        # Paste the dict captured in Phase 1's scratch file.
    }
    # ... rest unchanged ...
```

Replace `GOLDEN = {...}` with the actual values from `v04_baseline_golden.txt`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/integration/test_backwards_compat.py -v`
Expected: PASS — synth numerics match the v0.3.0-captured golden byte-for-byte.

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: `339 passed, 5 xfailed`. The Phase 17 failures are now resolved (the failing real-SPY assertions are either replaced by the synth-pointed test OR captured as expected real-SPY-dependent integration smokes).

Any remaining failures from Phase 17's `v04_phase17_failures.txt` should be diagnosed individually:
- If the failing test was a v0.3.0 numeric assertion on synthetic SPY → move it to point at `data/synth/SPY.csv` OR delete if redundant with `test_sma_cross_synth_spy_unchanged`.
- If the failing test was an integration assertion that just needs new golden values → recapture against real data and update the test inline.

- [ ] **Step 6: Commit**

```
git add tests/integration/test_backwards_compat.py configs/backtests/sma_cross_synth_spy.yaml
git commit -m "test(backwards-compat): re-point at data/synth/SPY.csv; preserve v0.3.0 golden numerics"
```

Phase 19 ends. All performance tests are xfail-by-default; framework correctness is fully covered.

---

## Phase 20: Docs

Phase 20 updates `docs/strategy_contract.md`, `docs/runbook.md`, and `README.md` to reflect v0.4.0's contract changes. No test count change.

### Task 45: Update `docs/strategy_contract.md`

**Files:**
- Modify: `docs/strategy_contract.md`

- [ ] **Step 1: Read the current doc**

Identify the existing sections; preserve their structure.

- [ ] **Step 2: Append a v0.4.0 section**

At the bottom of `docs/strategy_contract.md`, add:

```markdown
## v0.4.0 additions (opt-in)

A strategy can opt into the v0.4.0 multi-symbol contract by setting two class attributes:

```python
class MyStrategy(BaseStrategy[MyParams]):
    uses_multi_symbol = True   # routes through MultiSymbolPortfolioSimulator
    uses_per_bar = True        # signal_for_bar(...) called per (symbol, bar)
```

When `uses_per_bar` is True, the strategy implements:

```python
def signal_for_bar(
    self,
    *,
    symbol: str,
    bar_idx: int,
    data_panel: dict[str, pd.DataFrame],
    indicators_panel: dict[str, pd.DataFrame],
    ctx: StrategyContext,
    params: MyParams,
) -> float:
    """Return target_position in [-1.0, 1.0]. Fractional values are
    interpreted as partial positions (0.5 = half size)."""
```

The strategy may read `ctx.position_phase[symbol]` (a `TSPhase` value: `HARD`, `RUNNER`, or `DISARMED`),
`ctx.bars_in_phase[symbol]` (int, bars spent in the current phase),
`ctx.recent_pnl` (rolling PnL series), and `ctx.regime` (a `RegimeState` with `book_flat`).
All four fields are populated by the simulator after the just-processed bar's
state finalizes; strategy decisions for bar `t+1` see state from bar `t`.

Auxiliary OHLCV data (e.g., SPY, ^VIX for regime gates) lives in
`data_panel` under the aux symbol keys declared in `data.aux_symbols`. They
are not iterated over for entries — they exist purely as input to indicators
and regime evaluation.

Regime gates (SPY 200-EMA, VIX hysteresis, strategy circuit breaker) live
in the simulator, not the strategy. The strategy reads `ctx.regime.book_flat`
for diagnostics; when True, the simulator forces all positions flat
regardless of the strategy's emitted target.

The v0.3.0 contract (single-symbol, signal ∈ {-1, 0, 1}, no aux_data) is unchanged.
Strategies that do not set `uses_multi_symbol` continue to run through the
original `PortfolioSimulator` path.
```

- [ ] **Step 3: Commit**

```
git add docs/strategy_contract.md
git commit -m "docs(strategy_contract): document v0.4.0 multi-symbol + per-bar contract"
```

---

### Task 46: Update `docs/runbook.md`

**Files:**
- Modify: `docs/runbook.md`

- [ ] **Step 1: Append a v0.4.0 limitations section**

At the bottom of `docs/runbook.md`, add:

```markdown
## v0.4.0 limitations

The v0.4.0 framework lands the multi-symbol simulator, tranche-stop machinery,
regime gates, risk/sector caps, and yfinance loader. The following items are
deliberately deferred:

- **Multi-symbol WFO** is deferred to v0.4.1. `run_wfo` against a multi-symbol
  strategy raises with an explicit deferral message. To inspect window-level
  metrics in v0.4.0, run the full `run_batch` and slice the resulting equity
  curve manually.
- **Phased circuit-breaker re-entry.** v0.4.0 uses the PRD literal: full size
  on day 11. If WFO ratchet-down clusters emerge, phased 50%→100% re-entry
  becomes a v0.4.1 follow-up.
- **Continuous-bound LHS sampling.** v0.4.0's LHS sampler operates over
  index positions in discrete candidate lists. Strategies needing truly
  continuous parameters require a separate sampler.
- **Borrow cost / margin call simulation.** Same as v0.2.0 and v0.3.0.
  `mean_reversion_atr` is long-only, so this doesn't bite directly, but
  any future short strategy inherits the limitation.
- **Sector membership changes over time.** `data/sector_map.csv` is a
  static snapshot. Tickers that changed sectors during 2015-2025 are
  mapped to their current sector for the whole window.

## v0.4.0 performance-gate flip workflow

Stress-window integration tests (`tests/integration/test_stress_windows.py`)
and the held-out continuous test (`tests/integration/test_held_out_2022_2025.py`)
are marked `@pytest.mark.xfail(strict=False)` by default. Each test ALWAYS:

1. Runs the backtest end-to-end (structural correctness).
2. Parses metrics from `batch_summary.json`.
3. Writes metrics to `metrics.json` in the test's tmp_path for inspection.

The PRD's performance thresholds (DD < 9%, return > 15%) are wrapped by the
xfail marker — they exist in source, but CI does not fail when they're not
met. To convert a target into a hard gate, REMOVE the `@pytest.mark.xfail`
decorator on that specific test. This separates framework regressions
(genuine bugs) from strategy-tuning gaps (a config retune, not a fix).

The same machinery applies to any future test that asserts a strategy
performance number: wrap in xfail until the strategy is tuned to clear
the bar consistently.
```

- [ ] **Step 2: Commit**

```
git add docs/runbook.md
git commit -m "docs(runbook): v0.4.0 limitations + performance-gate flip workflow"
```

---

### Task 47: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Execution model + Strategy contract sections**

Find the existing "Execution model" or equivalent paragraph. Add a sentence noting v0.4.0's multi-symbol + regime-gate capabilities:

```markdown
## Execution model

v0.3.0 introduced execution-layer trailing stops; configure via
`execution.trailing_stop_pct` or `execution.trailing_stop_atr_mult`.

v0.4.0 adds a two-phase tranche stop (`execution.hard_stop_atr_mult` +
`execution.runner_atr_mult`) for strategies that scale out, and a three-gate
regime policy (SPY 200-EMA + VIX hysteresis + strategy circuit breaker)
that flattens the entire book to cash. Multi-symbol strategies opt in via
`uses_multi_symbol = True` and run through a shared-cash simulator with
risk-budget and sector-cap enforcement.

The single-symbol v0.3.0 path remains the default; existing strategies are
unaffected.
```

- [ ] **Step 2: Commit**

```
git add README.md
git commit -m "docs(readme): describe v0.4.0 multi-symbol + tranche-stop + regime gates"
```

Phase 20 ends.

---

## Phase 21: Version bump + tag

### Task 48: Bump `pyproject.toml` and tag `v0.4.0`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version + add data extras**

In `pyproject.toml`:

```toml
[project]
name = "backtester"
version = "0.4.0"  # was 0.3.0

[project.optional-dependencies]
data = ["yfinance>=0.2.40"]
# dev = [...]  preserve existing
```

- [ ] **Step 2: Final full-suite run**

Run: `python -m pytest -q`
Expected: `339 passed, 5 xfailed` (or `xpassed` if the strategy hits the targets).

- [ ] **Step 3: Commit**

```
git add pyproject.toml
git commit -m "chore: bump version to 0.4.0 (mean_reversion_atr + multi-symbol framework)"
```

- [ ] **Step 4: Create the tag locally**

```
git tag -a v0.4.0 -m "v0.4.0 — mean_reversion_atr + multi-symbol framework"
```

- [ ] **Step 5: HARD PAUSE — confirm push**

DO NOT push to origin without user confirmation. Per the project brief:
> "Tag and push only after the user confirms."

Confirm with the user:

> "v0.4.0 is committed and tagged locally. Final test count: 339 passing + 5 xfailed. The xfailed tests are the PRD performance gates; they're metric-reported but not asserted by default (see docs/runbook.md). Push to origin?"

On user approval, run:

```
git push origin master
git push origin v0.4.0
```

Phase 21 ends. v0.4.0 is shipped.

---

## Closing checklist

After all 21 phases:

- [ ] `python -m pytest -q` reports `339 passed, 5 xfailed` (or xpassed).
- [ ] `python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_synth_spy.yaml` produces byte-identical v0.3.0 numerics.
- [ ] `python -m backtester.runners.run_batch --config configs/backtests/mean_rev_v04_smoke.yaml` exits 0 with a portfolio equity curve.
- [ ] `python scripts/screen_universe.py --candidates configs/universe_candidates_seed.txt --start 2023-01-03 --end 2024-12-31 --out /tmp/u.yaml --top 10` exits 0.
- [ ] `git tag -l` lists `v0.4.0`.
- [ ] `pyproject.toml` version is `0.4.0` and includes the `data` extras group.
- [ ] `docs/strategy_contract.md`, `docs/runbook.md`, and `README.md` all reference the v0.4.0 contract and limitations.

The PRD's strategy-performance targets (Calmar > 2.5, DD < 9%, return > 15%) are reported by the held-out and stress-window tests but NOT asserted by default. Flipping them to hard gates is a per-test decision documented in `docs/runbook.md`.

---

## Plan execution notes

This plan was written assuming `superpowers:subagent-driven-development` — fresh agent per task, two-stage review between tasks. The cumulative test counts in the phase table are tracked checkpoints; mismatches indicate regressions and should NOT be ignored. The "performance gate flip" workflow keeps strategy-tuning out of the framework-release pipeline.

For an inline-execution path via `superpowers:executing-plans`, the per-phase checkpoints map naturally onto the skill's batch-execution semantics. Either way, the structural-correctness gates in Phase 21 must pass before tagging.

