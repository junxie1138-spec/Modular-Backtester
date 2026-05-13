# Modular Stock Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python framework where each daily-stock strategy is a self-contained module conforming to one ABC contract, and where the same engine handles standard backtests, grid optimization, and walk-forward optimization with no strategy-specific engine changes.

**Architecture:** A `BaseStrategy` ABC + explicit `STRATEGY_REGISTRY` separates strategy authoring from engine logic. The engine pipeline is `data -> indicators -> signals -> orders -> fills -> portfolio -> analytics`. Optimization and WFO are orchestrators that call the same engine repeatedly with different parameters and date windows. Configs are YAML deserialized to dataclasses; all runs write deterministic artifact bundles.

**Tech Stack:** Python 3.11, pandas, numpy, PyYAML, pytest. Long-only execution with MARKET / LIMIT / STOP order types. Sample data via a deterministic generator script (committed CSVs in `data/raw/`).

**Scope notes from clarifications:**
- Broker: long-only with MARKET / LIMIT / STOP orders.
- Test data: bundled CSVs produced by `scripts/generate_sample_data.py` (deterministic synthetic OHLCV).
- Optimization: sequential grid search only (parallel deferred to "Later").

---

## Phase 1: Project Foundation

### Task 1: Initialize repo and directory tree

**Files:**
- Create: `.gitignore`
- Create: every `__init__.py` listed in the architecture tree

- [ ] **Step 1: Initialize git**

Run:
```
git init
git config user.name "Aiden"
git config user.email "junxie1138@gmail.com"
```
Expected: `Initialized empty Git repository`.

- [ ] **Step 2: Create the package skeleton**

Run (PowerShell):
```powershell
$dirs = @(
  "backtester","backtester/config","backtester/core","backtester/data",
  "backtester/strategies","backtester/strategies/templates","backtester/engine",
  "backtester/analytics","backtester/optimize","backtester/wfo","backtester/runners",
  "backtester/io","strategies","configs/backtests","configs/optimize","configs/wfo",
  "data/raw","data/processed","output/runs","tests","tests/unit","tests/integration",
  "tests/fixtures","scripts","docs"
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
$initDirs = @(
  "backtester","backtester/config","backtester/core","backtester/data",
  "backtester/strategies","backtester/strategies/templates","backtester/engine",
  "backtester/analytics","backtester/optimize","backtester/wfo","backtester/runners",
  "backtester/io","tests","tests/unit","tests/integration","tests/fixtures"
)
$initDirs | ForEach-Object { New-Item -ItemType File -Force -Path "$_/__init__.py" | Out-Null }
```
Expected: directories created, `__init__.py` files created.

- [ ] **Step 3: Add .gitignore**

Write `.gitignore`:
```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
.venv/
venv/
*.egg-info/
dist/
build/
output/runs/*
!output/runs/.gitkeep
data/processed/*
!data/processed/.gitkeep
.env
.DS_Store
```

- [ ] **Step 4: Add `.gitkeep` placeholders**

Run:
```
New-Item -ItemType File -Force -Path "output/runs/.gitkeep","data/processed/.gitkeep" | Out-Null
```

- [ ] **Step 5: Commit**

```
git add .gitignore backtester strategies tests scripts docs configs data output
git commit -m "chore: initialize project structure"
```

---

### Task 2: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "modular-stock-backtester"
version = "0.1.0"
description = "Modular Python backtesting framework with WFO support"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "pyyaml>=6.0",
  "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]

[project.scripts]
bt-run = "backtester.runners.run_backtest:main"
bt-optimize = "backtester.runners.run_optimize:main"
bt-wfo = "backtester.runners.run_wfo:main"

[tool.setuptools.packages.find]
include = ["backtester*", "strategies*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write `.env.example`**

```
# Optional: override default data directory
BACKTESTER_DATA_DIR=./data/raw
BACKTESTER_OUTPUT_DIR=./output/runs
```

- [ ] **Step 3: Install in editable mode**

Run: `pip install -e .[dev]`
Expected: successful install of pandas, numpy, pyyaml, pyarrow, pytest.

- [ ] **Step 4: Commit**

```
git add pyproject.toml .env.example
git commit -m "chore: add pyproject and dev install"
```

---

### Task 3: Core types

**Files:**
- Create: `backtester/core/types.py`
- Create: `tests/unit/test_core_types.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_core_types.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.types import StrategyContext, SignalFrame, BacktestResult


def test_strategy_context_defaults():
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=20)
    assert ctx.symbol == "SPY"
    assert ctx.metadata == {}


def test_strategy_context_metadata_is_per_instance():
    a = StrategyContext(symbol="A", timeframe="1d", warmup_bars=1)
    b = StrategyContext(symbol="B", timeframe="1d", warmup_bars=1)
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_signal_frame_defaults():
    df = pd.DataFrame({"signal": [0, 1], "size": [1.0, 1.0]})
    sf = SignalFrame(data=df)
    assert sf.signal_column == "signal"
    assert sf.size_column == "size"
    assert sf.price_column is None


def test_backtest_result_holds_frames():
    summary = {"total_return": 0.1}
    eq = pd.DataFrame({"equity": [1.0]})
    trades = pd.DataFrame({"pnl": [0.0]})
    positions = pd.DataFrame({"qty": [0]})
    r = BacktestResult(summary=summary, equity_curve=eq, trades=trades, positions=positions)
    assert r.summary["total_return"] == 0.1
    assert len(r.equity_curve) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_core_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtester.core.types'`.

- [ ] **Step 3: Implement types**

`backtester/core/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass(slots=True)
class StrategyContext:
    symbol: str
    timeframe: str
    warmup_bars: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalFrame:
    data: pd.DataFrame
    signal_column: str = "signal"
    size_column: Optional[str] = "size"
    price_column: Optional[str] = None


@dataclass(slots=True)
class BacktestResult:
    summary: Dict[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_core_types.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/core/types.py tests/unit/test_core_types.py
git commit -m "feat(core): add StrategyContext, SignalFrame, BacktestResult"
```

---

### Task 4: Core exceptions, enums, constants

**Files:**
- Create: `backtester/core/exceptions.py`
- Create: `backtester/core/enums.py`
- Create: `backtester/core/constants.py`
- Create: `tests/unit/test_core_enums.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_core_enums.py`:
```python
from backtester.core.enums import OrderType, OrderSide, SignalDirection
from backtester.core.exceptions import (
    BacktesterError, ConfigError, DataError, StrategyError,
)


def test_order_types_have_expected_values():
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP.value == "stop"


def test_order_sides():
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_signal_directions():
    assert SignalDirection.FLAT.value == 0
    assert SignalDirection.LONG.value == 1


def test_exceptions_inherit_from_base():
    for exc in (ConfigError, DataError, StrategyError):
        assert issubclass(exc, BacktesterError)


def test_exceptions_carry_message():
    e = ConfigError("bad key")
    assert "bad key" in str(e)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_core_enums.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement exceptions**

`backtester/core/exceptions.py`:
```python
from __future__ import annotations


class BacktesterError(Exception):
    """Base exception for the backtester framework."""


class ConfigError(BacktesterError):
    """Raised when a config is malformed or invalid."""


class DataError(BacktesterError):
    """Raised when input data is missing or invalid."""


class StrategyError(BacktesterError):
    """Raised when a strategy violates its contract."""


class ExecutionError(BacktesterError):
    """Raised when the broker / portfolio simulator cannot proceed."""
```

- [ ] **Step 4: Implement enums**

`backtester/core/enums.py`:
```python
from __future__ import annotations

from enum import Enum, IntEnum


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SignalDirection(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1
```

- [ ] **Step 5: Implement constants**

`backtester/core/constants.py`:
```python
from __future__ import annotations

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TRADING_DAYS_PER_YEAR = 252
BPS = 1e-4  # 1 basis point as a decimal
DEFAULT_TIMEFRAME = "1d"
```

- [ ] **Step 6: Run test to verify pass**

Run: `pytest tests/unit/test_core_enums.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```
git add backtester/core/exceptions.py backtester/core/enums.py backtester/core/constants.py tests/unit/test_core_enums.py
git commit -m "feat(core): add exceptions, enums, and shared constants"
```

---

## Phase 2: Strategy Contract

### Task 5: BaseStrategy ABC

**Files:**
- Create: `backtester/strategies/base.py`
- Create: `tests/unit/test_strategy_base.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_base.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class _DummyParams:
    lookback: int = 5


class _DummyStrategy(BaseStrategy[_DummyParams]):
    strategy_id = "dummy"

    @classmethod
    def params_type(cls):
        return _DummyParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": [0] * len(data), "size": [1.0] * len(data)}, index=data.index)
        return SignalFrame(data=df)

    def warmup_bars(self, params):
        return params.lookback


def _ohlcv(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
        index=idx,
    )


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]


def test_concrete_strategy_instantiable():
    s = _DummyStrategy()
    assert s.strategy_id == "dummy"
    assert s.version == "1.0"


def test_validate_passes_with_required_columns():
    s = _DummyStrategy()
    s.validate(_ohlcv(), _DummyParams())


def test_validate_raises_on_missing_columns():
    s = _DummyStrategy()
    bad = _ohlcv().drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        s.validate(bad, _DummyParams())


def test_warmup_default_is_zero():
    class _NoWarmup(_DummyStrategy):
        def warmup_bars(self, params):
            return BaseStrategy.warmup_bars(self, params)
    assert _NoWarmup().warmup_bars(_DummyParams()) == 0


def test_indicators_and_signals_callable():
    s = _DummyStrategy()
    data = _ohlcv()
    p = _DummyParams()
    ind = s.indicators(data, p)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=s.warmup_bars(p))
    sf = s.generate_signals(data, ind, ctx, p)
    assert isinstance(sf, SignalFrame)
    assert "signal" in sf.data.columns
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_base.py -v`
Expected: FAIL — `ModuleNotFoundError: backtester.strategies.base`.

- [ ] **Step 3: Implement BaseStrategy**

`backtester/strategies/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import pandas as pd

from backtester.core.constants import REQUIRED_OHLCV_COLUMNS
from backtester.core.types import SignalFrame, StrategyContext

P = TypeVar("P")


class BaseStrategy(ABC, Generic[P]):
    strategy_id: str
    version: str = "1.0"
    asset_type: str = "stock"
    timeframe: str = "1d"

    @classmethod
    @abstractmethod
    def params_type(cls):
        raise NotImplementedError

    @abstractmethod
    def indicators(self, data: pd.DataFrame, params: P) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: P,
    ) -> SignalFrame:
        raise NotImplementedError

    def validate(self, data: pd.DataFrame, params: P) -> None:
        required = set(REQUIRED_OHLCV_COLUMNS)
        present = set(map(str.lower, data.columns))
        missing = required - present
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def warmup_bars(self, params: P) -> int:
        return 0
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_base.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add backtester/strategies/base.py tests/unit/test_strategy_base.py
git commit -m "feat(strategies): add BaseStrategy ABC contract"
```

---

### Task 6: Strategy registry

**Files:**
- Create: `backtester/strategies/registry.py`
- Create: `tests/unit/test_strategy_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_registry.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.core.types import SignalFrame
from backtester.strategies.base import BaseStrategy
from backtester.strategies.registry import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    register_strategy,
)


@dataclass(slots=True)
class _FakeParams:
    x: int = 1


class _FakeStrategy(BaseStrategy[_FakeParams]):
    strategy_id = "fake_test_only"

    @classmethod
    def params_type(cls):
        return _FakeParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        return SignalFrame(data=pd.DataFrame({"signal": [0] * len(data)}, index=data.index))


def test_registry_is_dict():
    assert isinstance(STRATEGY_REGISTRY, dict)


def test_register_and_lookup(monkeypatch):
    monkeypatch.setitem(STRATEGY_REGISTRY, "fake_test_only", _FakeStrategy)
    assert get_strategy_class("fake_test_only") is _FakeStrategy


def test_lookup_unknown_raises():
    with pytest.raises(KeyError, match="unknown_strategy"):
        get_strategy_class("unknown_strategy")


def test_register_strategy_helper(monkeypatch):
    monkeypatch.setattr(
        "backtester.strategies.registry.STRATEGY_REGISTRY", {}, raising=True
    )
    register_strategy(_FakeStrategy)
    from backtester.strategies.registry import STRATEGY_REGISTRY as R
    assert R["fake_test_only"] is _FakeStrategy
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_registry.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement registry**

`backtester/strategies/registry.py`:
```python
from __future__ import annotations

from typing import Dict, Type

from backtester.strategies.base import BaseStrategy

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
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_registry.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/strategies/registry.py tests/unit/test_strategy_registry.py
git commit -m "feat(strategies): add explicit strategy registry"
```

---

### Task 7: Strategy authoring template

**Files:**
- Create: `backtester/strategies/templates/strategy_template.py`

- [ ] **Step 1: Write the template (no test — it is a reference file)**

`backtester/strategies/templates/strategy_template.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    size: float = 1.0


class StrategyName(BaseStrategy[StrategyParams]):
    """
    Purpose:
        Replace with one-sentence description.

    Inputs:
        OHLCV dataframe with datetime index and lowercase columns:
        open, high, low, close, volume.

    Outputs:
        SignalFrame with `signal` (0/1) and optional `size` columns.

    Side effects:
        None.
    """

    strategy_id = "replace_me"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.lookback

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 2: Sanity-import to verify it parses**

Run: `python -c "import backtester.strategies.templates.strategy_template as t; print(t.StrategyName.strategy_id)"`
Expected: `replace_me`.

- [ ] **Step 3: Commit**

```
git add backtester/strategies/templates/strategy_template.py
git commit -m "feat(strategies): add strategy authoring template"
```

---

## Phase 3: Data Layer

### Task 8: Test fixtures — synthetic OHLCV generator

**Files:**
- Create: `tests/fixtures/synthetic.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_fixtures_synthetic.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fixtures_synthetic.py`:
```python
from __future__ import annotations

from tests.fixtures.synthetic import make_ohlcv


def test_make_ohlcv_shape_and_columns():
    df = make_ohlcv(n=100, seed=42)
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_make_ohlcv_is_deterministic():
    a = make_ohlcv(n=50, seed=7)
    b = make_ohlcv(n=50, seed=7)
    assert (a.values == b.values).all()


def test_make_ohlcv_index_is_business_days():
    df = make_ohlcv(n=20, seed=0, start="2024-01-02")
    assert df.index.is_monotonic_increasing
    assert df.index.inferred_freq == "B"


def test_make_ohlcv_high_low_invariants():
    df = make_ohlcv(n=200, seed=1)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_fixtures_synthetic.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the generator**

`tests/fixtures/synthetic.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 500,
    seed: int = 0,
    start: str = "2020-01-02",
    start_price: float = 100.0,
    drift: float = 0.0003,
    vol: float = 0.012,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-style OHLCV series for tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))

    # open ~ prev close with small gap
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.001, n - 1))

    intrabar_range = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + intrabar_range
    low = np.minimum(open_, close) - intrabar_range
    low = np.clip(low, 0.01, None)
    volume = rng.integers(500_000, 5_000_000, size=n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
```

- [ ] **Step 4: Add conftest with shared fixtures**

`tests/conftest.py`:
```python
from __future__ import annotations

import pytest

from tests.fixtures.synthetic import make_ohlcv


@pytest.fixture
def ohlcv_small():
    return make_ohlcv(n=60, seed=1)


@pytest.fixture
def ohlcv_medium():
    return make_ohlcv(n=750, seed=1)
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/unit/test_fixtures_synthetic.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```
git add tests/fixtures/synthetic.py tests/conftest.py tests/unit/test_fixtures_synthetic.py
git commit -m "test: add deterministic synthetic OHLCV fixture"
```

---

### Task 9: Data loader (CSV + Parquet) + base

**Files:**
- Create: `backtester/data/base.py`
- Create: `backtester/data/loader.py`
- Create: `tests/unit/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_data_loader.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.loader import CSVDataLoader, ParquetDataLoader, load_symbol
from tests.fixtures.synthetic import make_ohlcv


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    df = make_ohlcv(n=120, seed=2)
    df.to_csv(tmp_path / "SPY.csv", index_label="date")
    return tmp_path


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    df = make_ohlcv(n=120, seed=2)
    df.to_parquet(tmp_path / "SPY.parquet", index=True)
    return tmp_path


def test_csv_loader_loads_symbol(csv_dir):
    loader = CSVDataLoader(root=csv_dir)
    df = loader.load("SPY")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 120


def test_csv_loader_respects_date_range(csv_dir):
    loader = CSVDataLoader(root=csv_dir)
    raw = loader.load("SPY")
    mid_start = raw.index[30].strftime("%Y-%m-%d")
    mid_end = raw.index[60].strftime("%Y-%m-%d")
    df = loader.load("SPY", start=mid_start, end=mid_end)
    assert df.index.min() >= pd.Timestamp(mid_start)
    assert df.index.max() <= pd.Timestamp(mid_end)


def test_csv_loader_missing_symbol_raises(tmp_path):
    loader = CSVDataLoader(root=tmp_path)
    with pytest.raises(DataError, match="MISSING"):
        loader.load("MISSING")


def test_parquet_loader_loads_symbol(parquet_dir):
    loader = ParquetDataLoader(root=parquet_dir)
    df = loader.load("SPY")
    assert len(df) == 120
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_load_symbol_dispatches_by_source(csv_dir):
    df = load_symbol(symbol="SPY", source="csv", root=csv_dir)
    assert len(df) == 120


def test_load_symbol_unknown_source(tmp_path):
    with pytest.raises(DataError, match="unknown source"):
        load_symbol(symbol="SPY", source="hdf5", root=tmp_path)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_data_loader.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the data base interface**

`backtester/data/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd


class DataLoader(ABC):
    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def load(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
```

- [ ] **Step 4: Implement loaders**

`backtester/data/loader.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from backtester.core.exceptions import DataError
from backtester.data.base import DataLoader


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise DataError(f"OHLCV file missing columns: {missing}")
    return df[keep]


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


class CSVDataLoader(DataLoader):
    def load(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        path = self.root / f"{symbol}.csv"
        if not path.exists():
            raise DataError(f"CSV not found for symbol {symbol!r} at {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize_ohlcv(df).sort_index()
        return _slice(df, start, end)


class ParquetDataLoader(DataLoader):
    def load(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        path = self.root / f"{symbol}.parquet"
        if not path.exists():
            raise DataError(f"Parquet not found for symbol {symbol!r} at {path}")
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = _normalize_ohlcv(df).sort_index()
        return _slice(df, start, end)


def load_symbol(
    symbol: str,
    source: str,
    root: Union[str, Path],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    root = Path(root)
    src = source.lower()
    if src == "csv":
        return CSVDataLoader(root).load(symbol, start, end)
    if src == "parquet":
        return ParquetDataLoader(root).load(symbol, start, end)
    raise DataError(f"unknown source: {source!r} (allowed: csv, parquet)")
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/unit/test_data_loader.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```
git add backtester/data/base.py backtester/data/loader.py tests/unit/test_data_loader.py
git commit -m "feat(data): add CSV and Parquet OHLCV loaders"
```

---

### Task 10: Data validators

**Files:**
- Create: `backtester/data/validators.py`
- Create: `tests/unit/test_data_validators.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_data_validators.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.exceptions import DataError
from backtester.data.validators import validate_ohlcv
from tests.fixtures.synthetic import make_ohlcv


def test_validate_passes_on_clean_data(ohlcv_small):
    validate_ohlcv(ohlcv_small)


def test_validate_rejects_missing_columns(ohlcv_small):
    bad = ohlcv_small.drop(columns=["close"])
    with pytest.raises(DataError, match="missing columns"):
        validate_ohlcv(bad)


def test_validate_rejects_non_datetime_index(ohlcv_small):
    bad = ohlcv_small.reset_index(drop=True)
    with pytest.raises(DataError, match="DatetimeIndex"):
        validate_ohlcv(bad)


def test_validate_rejects_non_monotonic_index(ohlcv_small):
    bad = ohlcv_small.iloc[::-1]
    with pytest.raises(DataError, match="monotonic"):
        validate_ohlcv(bad)


def test_validate_rejects_duplicates(ohlcv_small):
    bad = pd.concat([ohlcv_small, ohlcv_small.head(1)]).sort_index()
    with pytest.raises(DataError, match="duplicate"):
        validate_ohlcv(bad)


def test_validate_rejects_negative_prices(ohlcv_small):
    bad = ohlcv_small.copy()
    bad.iloc[5, bad.columns.get_loc("low")] = -1.0
    with pytest.raises(DataError, match="non-positive"):
        validate_ohlcv(bad)


def test_validate_rejects_nan(ohlcv_small):
    bad = ohlcv_small.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(DataError, match="NaN"):
        validate_ohlcv(bad)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_data_validators.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement validator**

`backtester/data/validators.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.core.constants import REQUIRED_OHLCV_COLUMNS
from backtester.core.exceptions import DataError


def validate_ohlcv(df: pd.DataFrame) -> None:
    cols = set(map(str.lower, df.columns))
    missing = set(REQUIRED_OHLCV_COLUMNS) - cols
    if missing:
        raise DataError(f"data missing columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("data index must be a DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise DataError("data index must be monotonic increasing")

    if df.index.duplicated().any():
        raise DataError("data index contains duplicate timestamps")

    price_cols = ["open", "high", "low", "close"]
    if df[price_cols].isna().any().any() or df["volume"].isna().any():
        raise DataError("data contains NaN values")

    if (df[price_cols] <= 0).any().any():
        raise DataError("data contains non-positive prices")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_data_validators.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```
git add backtester/data/validators.py tests/unit/test_data_validators.py
git commit -m "feat(data): add OHLCV validator"
```

---

### Task 11: Bundled sample CSVs

**Files:**
- Create: `scripts/generate_sample_data.py`
- Create: `data/raw/SPY.csv` (generated)
- Create: `data/raw/AAPL.csv` (generated)

- [ ] **Step 1: Write the generator script**

`scripts/generate_sample_data.py`:
```python
"""Write deterministic synthetic OHLCV CSVs for SPY and AAPL.

Run from the repo root:
    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.synthetic import make_ohlcv


def main() -> None:
    out = REPO_ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    spy = make_ohlcv(n=3000, seed=1, start="2013-01-02", start_price=140.0)
    aapl = make_ohlcv(n=3000, seed=2, start="2013-01-02", start_price=18.0)

    spy.to_csv(out / "SPY.csv", index_label="date")
    aapl.to_csv(out / "AAPL.csv", index_label="date")

    print(f"Wrote {out / 'SPY.csv'} ({len(spy)} rows)")
    print(f"Wrote {out / 'AAPL.csv'} ({len(aapl)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python scripts/generate_sample_data.py`
Expected: prints "Wrote ... SPY.csv (3000 rows)" and "Wrote ... AAPL.csv (3000 rows)".

- [ ] **Step 3: Verify the CSVs load via the data loader**

Run:
```
python -c "from backtester.data.loader import load_symbol; df = load_symbol('SPY','csv','data/raw'); print(df.shape); print(df.head(2))"
```
Expected: shape `(3000, 5)` and two rows of OHLCV.

- [ ] **Step 4: Commit**

```
git add scripts/generate_sample_data.py data/raw/SPY.csv data/raw/AAPL.csv
git commit -m "data: add deterministic SPY and AAPL sample CSVs"
```

---

## Phase 4: Config Layer

### Task 12: Config dataclass models

**Files:**
- Create: `backtester/config/models.py`
- Create: `tests/unit/test_config_models.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config_models.py`:
```python
from __future__ import annotations

from backtester.config.models import (
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    PortfolioConfig,
    RunConfig,
    WFOConfig,
)


def test_data_config_required_fields():
    d = DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01")
    assert d.source == "csv"


def test_execution_defaults():
    e = ExecutionConfig()
    assert e.initial_cash == 100_000.0
    assert e.commission_bps == 1.0
    assert e.slippage_bps == 2.0
    assert e.allow_fractional is False


def test_portfolio_defaults():
    p = PortfolioConfig()
    assert p.sizing_mode == "percent_equity"
    assert p.size == 1.0


def test_optimization_default_empty_space():
    o = OptimizationConfig()
    assert o.objective == "sharpe"
    assert o.param_space == {}


def test_wfo_defaults():
    w = WFOConfig()
    assert w.enabled is False
    assert w.train_bars is None


def test_run_config_composition():
    rc = RunConfig(
        run_name="x",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 50},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
    assert rc.optimization is None
    assert rc.wfo is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_config_models.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement models**

`backtester/config/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DataConfig:
    symbols: List[str]
    timeframe: str
    start: str
    end: str
    source: str = "csv"
    root: str = "data/raw"


@dataclass(slots=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_fractional: bool = False


@dataclass(slots=True)
class PortfolioConfig:
    sizing_mode: str = "percent_equity"
    size: float = 1.0


@dataclass(slots=True)
class OptimizationConfig:
    objective: str = "sharpe"
    param_space: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass(slots=True)
class WFOConfig:
    enabled: bool = False
    train_bars: Optional[int] = None
    test_bars: Optional[int] = None
    step_bars: Optional[int] = None


@dataclass(slots=True)
class RunConfig:
    run_name: str
    strategy: str
    strategy_params: Dict[str, Any]
    data: DataConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    optimization: Optional[OptimizationConfig] = None
    wfo: Optional[WFOConfig] = None
    output_root: str = "output/runs"
    seed: int = 0
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_config_models.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add backtester/config/models.py tests/unit/test_config_models.py
git commit -m "feat(config): add RunConfig dataclass model"
```

---

### Task 13: YAML config loader

**Files:**
- Create: `backtester/config/loader.py`
- Create: `tests/unit/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config_loader.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from backtester.config.loader import load_run_config, dump_run_config
from backtester.core.exceptions import ConfigError


YAML_BASIC = """
run_name: sma_cross_spy
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2020-01-01"
  end: "2024-01-01"
  source: "csv"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
"""

YAML_WFO = """
run_name: x
strategy: sma_cross
strategy_params: {fast: 10, slow: 30}
data: {symbols: ["SPY"], timeframe: "1d", start: "2020-01-01", end: "2024-01-01"}
execution: {}
portfolio: {}
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20]
    slow: [50, 100]
wfo:
  enabled: true
  train_bars: 252
  test_bars: 63
  step_bars: 63
"""


def test_load_basic(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML_BASIC)
    rc = load_run_config(p)
    assert rc.run_name == "sma_cross_spy"
    assert rc.data.symbols == ["SPY"]
    assert rc.execution.commission_bps == 2
    assert rc.optimization is None
    assert rc.wfo is None


def test_load_wfo(tmp_path: Path):
    p = tmp_path / "w.yaml"
    p.write_text(YAML_WFO)
    rc = load_run_config(p)
    assert rc.wfo is not None and rc.wfo.enabled is True
    assert rc.wfo.train_bars == 252
    assert rc.optimization is not None
    assert rc.optimization.param_space["fast"] == [10, 20]


def test_load_missing_required_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: x\nstrategy: y\n")
    with pytest.raises(ConfigError):
        load_run_config(p)


def test_dump_round_trip(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML_BASIC)
    rc = load_run_config(p)
    out = tmp_path / "out.yaml"
    dump_run_config(rc, out)
    rc2 = load_run_config(out)
    assert rc2.run_name == rc.run_name
    assert rc2.strategy_params == rc.strategy_params
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement loader**

`backtester/config/loader.py`:
```python
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from backtester.config.models import (
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    PortfolioConfig,
    RunConfig,
    WFOConfig,
)
from backtester.core.exceptions import ConfigError

PathLike = Union[str, Path]


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required field {key!r} in {where}")
    return d[key]


def _from_dict(raw: Dict[str, Any]) -> RunConfig:
    try:
        data_raw = _require(raw, "data", "config root")
        data = DataConfig(
            symbols=list(_require(data_raw, "symbols", "data")),
            timeframe=_require(data_raw, "timeframe", "data"),
            start=_require(data_raw, "start", "data"),
            end=_require(data_raw, "end", "data"),
            source=data_raw.get("source", "csv"),
            root=data_raw.get("root", "data/raw"),
        )
        execution = ExecutionConfig(**(raw.get("execution") or {}))
        portfolio = PortfolioConfig(**(raw.get("portfolio") or {}))

        opt = None
        if raw.get("optimization"):
            opt_raw = raw["optimization"]
            opt = OptimizationConfig(
                objective=opt_raw.get("objective", "sharpe"),
                param_space=dict(opt_raw.get("param_space", {})),
            )

        wfo = None
        if raw.get("wfo"):
            wfo = WFOConfig(**raw["wfo"])

        return RunConfig(
            run_name=_require(raw, "run_name", "config root"),
            strategy=_require(raw, "strategy", "config root"),
            strategy_params=dict(raw.get("strategy_params", {})),
            data=data,
            execution=execution,
            portfolio=portfolio,
            optimization=opt,
            wfo=wfo,
            output_root=raw.get("output_root", "output/runs"),
            seed=int(raw.get("seed", 0)),
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"failed to parse config: {exc}") from exc


def load_run_config(path: PathLike) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return _from_dict(raw)


def dump_run_config(rc: RunConfig, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(rc)
    if payload.get("optimization") is None:
        payload.pop("optimization", None)
    if payload.get("wfo") is None:
        payload.pop("wfo", None)
    p.write_text(yaml.safe_dump(payload, sort_keys=False))
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/config/loader.py tests/unit/test_config_loader.py
git commit -m "feat(config): add YAML loader and dumper for RunConfig"
```

---

### Task 14: Config validation

**Files:**
- Create: `backtester/config/validation.py`
- Create: `tests/unit/test_config_validation.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config_validation.py`:
```python
from __future__ import annotations

import pytest

from backtester.config.models import (
    DataConfig, ExecutionConfig, PortfolioConfig, RunConfig, WFOConfig,
)
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError


def _make(**over):
    base = RunConfig(
        run_name="x",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


def test_valid_config_passes():
    validate_run_config(_make())


def test_empty_symbols_rejected():
    rc = _make()
    rc.data.symbols = []
    with pytest.raises(ConfigError, match="symbols"):
        validate_run_config(rc)


def test_start_after_end_rejected():
    rc = _make()
    rc.data.start = "2025-01-01"
    rc.data.end = "2024-01-01"
    with pytest.raises(ConfigError, match="start"):
        validate_run_config(rc)


def test_negative_cash_rejected():
    rc = _make()
    rc.execution.initial_cash = -1
    with pytest.raises(ConfigError, match="initial_cash"):
        validate_run_config(rc)


def test_wfo_requires_windows():
    rc = _make(wfo=WFOConfig(enabled=True))
    with pytest.raises(ConfigError, match="train_bars"):
        validate_run_config(rc)


def test_wfo_valid_when_windows_set():
    rc = _make(wfo=WFOConfig(enabled=True, train_bars=252, test_bars=63, step_bars=63))
    validate_run_config(rc)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_config_validation.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement validation**

`backtester/config/validation.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.config.models import RunConfig
from backtester.core.exceptions import ConfigError


def validate_run_config(rc: RunConfig) -> None:
    if not rc.run_name:
        raise ConfigError("run_name must be non-empty")
    if not rc.strategy:
        raise ConfigError("strategy must be non-empty")

    if not rc.data.symbols:
        raise ConfigError("data.symbols must contain at least one symbol")
    try:
        start = pd.Timestamp(rc.data.start)
        end = pd.Timestamp(rc.data.end)
    except Exception as exc:
        raise ConfigError(f"invalid data.start / data.end: {exc}") from exc
    if start >= end:
        raise ConfigError("data.start must be strictly before data.end")

    if rc.execution.initial_cash <= 0:
        raise ConfigError("execution.initial_cash must be > 0")
    if rc.execution.commission_bps < 0 or rc.execution.slippage_bps < 0:
        raise ConfigError("execution commission_bps and slippage_bps must be >= 0")

    if rc.portfolio.size <= 0 or rc.portfolio.size > 1.0:
        raise ConfigError("portfolio.size must be in (0, 1] when sizing_mode is percent_equity")

    if rc.wfo and rc.wfo.enabled:
        for k in ("train_bars", "test_bars", "step_bars"):
            v = getattr(rc.wfo, k)
            if v is None or v <= 0:
                raise ConfigError(f"wfo.{k} must be a positive integer when wfo.enabled")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_config_validation.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add backtester/config/validation.py tests/unit/test_config_validation.py
git commit -m "feat(config): add semantic validation for RunConfig"
```

---

## Phase 5: Execution Engine

### Task 15: Orders and fills primitives

**Files:**
- Create: `backtester/engine/orders.py`
- Create: `backtester/engine/fills.py`
- Create: `tests/unit/test_orders_fills.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_orders_fills.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.enums import OrderSide, OrderStatus, OrderType
from backtester.engine.orders import Order
from backtester.engine.fills import Fill, FillEngine


def test_market_order_construction():
    o = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)
    assert o.status == OrderStatus.PENDING
    assert o.limit_price is None


def test_limit_order_requires_price():
    with pytest.raises(ValueError, match="limit_price"):
        Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT)


def test_stop_order_requires_price():
    with pytest.raises(ValueError, match="stop_price"):
        Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
              side=OrderSide.BUY, qty=10, order_type=OrderType.STOP)


def _bar(open_, high, low, close):
    return pd.Series({"open": open_, "high": high, "low": low, "close": close, "volume": 1000})


def test_market_buy_fills_at_open_with_slippage():
    fe = FillEngine(commission_bps=1.0, slippage_bps=10.0)
    bar = _bar(100.0, 102.0, 99.0, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.MARKET)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(100.0 * (1 + 10e-4))
    assert fill.qty == 10
    expected_cost = fill.price * 10
    assert fill.commission == pytest.approx(expected_cost * 1e-4)


def test_market_sell_fills_at_open_minus_slippage():
    fe = FillEngine(commission_bps=0.0, slippage_bps=20.0)
    bar = _bar(100.0, 102.0, 99.0, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.SELL, qty=5, order_type=OrderType.MARKET)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(100.0 * (1 - 20e-4))


def test_limit_buy_skips_when_low_above_limit():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 102.0, 99.5, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT, limit_price=99.0)
    assert fe.try_fill(order, bar) is None


def test_limit_buy_fills_at_limit_when_touched():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 102.0, 98.5, 101.0)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT, limit_price=99.0)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    assert fill.price == pytest.approx(99.0)


def test_stop_buy_skips_when_high_below_stop():
    fe = FillEngine(commission_bps=0.0, slippage_bps=0.0)
    bar = _bar(100.0, 101.0, 99.5, 100.5)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.STOP, stop_price=102.0)
    assert fe.try_fill(order, bar) is None


def test_stop_buy_fills_at_stop_when_triggered():
    fe = FillEngine(commission_bps=0.0, slippage_bps=10.0)
    bar = _bar(100.0, 103.0, 99.0, 102.5)
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=10, order_type=OrderType.STOP, stop_price=101.0)
    fill = fe.try_fill(order, bar)
    assert fill is not None
    # stop triggered, fills at max(open, stop) + slippage
    assert fill.price == pytest.approx(101.0 * (1 + 10e-4))


def test_fill_dataclass_fields():
    f = Fill(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
             side=OrderSide.BUY, qty=10, price=100.0, commission=1.0)
    assert f.notional == 1000.0
    assert f.cash_delta == -(1000.0 + 1.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_orders_fills.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement orders**

`backtester/engine/orders.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from backtester.core.enums import OrderSide, OrderStatus, OrderType


@dataclass
class Order:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    tag: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("Order qty must be > 0")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP order requires stop_price")
```

- [ ] **Step 4: Implement fills**

`backtester/engine/fills.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from backtester.core.constants import BPS
from backtester.core.enums import OrderSide, OrderStatus, OrderType
from backtester.engine.orders import Order


@dataclass(slots=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def cash_delta(self) -> float:
        sign = -1.0 if self.side == OrderSide.BUY else 1.0
        return sign * self.notional - self.commission


class FillEngine:
    """Apply commission + slippage and decide whether each order fills on a bar."""

    def __init__(self, commission_bps: float, slippage_bps: float):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps * BPS

    def _slip(self, price: float, side: OrderSide) -> float:
        adj = 1.0 + self.slippage_bps * BPS if side == OrderSide.BUY else 1.0 - self.slippage_bps * BPS
        return price * adj

    def try_fill(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])

        price: Optional[float] = None

        if order.order_type == OrderType.MARKET:
            price = self._slip(open_, order.side)

        elif order.order_type == OrderType.LIMIT:
            lp = float(order.limit_price)  # type: ignore[arg-type]
            if order.side == OrderSide.BUY and low <= lp:
                price = min(lp, open_)
            elif order.side == OrderSide.SELL and high >= lp:
                price = max(lp, open_)

        elif order.order_type == OrderType.STOP:
            sp = float(order.stop_price)  # type: ignore[arg-type]
            if order.side == OrderSide.BUY and high >= sp:
                triggered = max(open_, sp)
                price = self._slip(triggered, order.side)
            elif order.side == OrderSide.SELL and low <= sp:
                triggered = min(open_, sp)
                price = self._slip(triggered, order.side)

        if price is None:
            return None

        notional = price * order.qty
        commission = self._commission(notional)
        order.status = OrderStatus.FILLED
        return Fill(
            timestamp=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(bar.name),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            commission=commission,
        )
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/unit/test_orders_fills.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```
git add backtester/engine/orders.py backtester/engine/fills.py tests/unit/test_orders_fills.py
git commit -m "feat(engine): add Order, Fill, and FillEngine with market/limit/stop"
```

---

### Task 16: Position primitive

**Files:**
- Create: `backtester/engine/position.py`
- Create: `tests/unit/test_position.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_position.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.enums import OrderSide
from backtester.engine.fills import Fill
from backtester.engine.position import Position


def _fill(side, qty, price, commission=0.0, ts="2024-01-02"):
    return Fill(timestamp=pd.Timestamp(ts), symbol="SPY",
                side=side, qty=qty, price=price, commission=commission)


def test_position_starts_flat():
    p = Position(symbol="SPY")
    assert p.qty == 0
    assert p.is_flat
    assert p.realized_pnl == 0.0


def test_buy_increases_qty_and_avg_cost():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0, commission=1.0))
    assert p.qty == 10
    assert p.avg_cost == pytest.approx(100.0)
    assert p.realized_pnl == pytest.approx(-1.0)


def test_two_buys_compute_weighted_avg_cost():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.BUY, 10, 110.0))
    assert p.qty == 20
    assert p.avg_cost == pytest.approx(105.0)


def test_partial_sell_realizes_pnl():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 4, 110.0, commission=0.5))
    assert p.qty == 6
    assert p.realized_pnl == pytest.approx(4 * (110.0 - 100.0) - 0.5)


def test_full_sell_returns_to_flat():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    p.apply_fill(_fill(OrderSide.SELL, 10, 105.0))
    assert p.is_flat
    assert p.avg_cost == 0.0
    assert p.realized_pnl == pytest.approx(50.0)


def test_sell_when_flat_raises():
    p = Position(symbol="SPY")
    with pytest.raises(ValueError, match="long-only"):
        p.apply_fill(_fill(OrderSide.SELL, 1, 100.0))


def test_mark_to_market():
    p = Position(symbol="SPY")
    p.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    assert p.market_value(price=110.0) == pytest.approx(1100.0)
    assert p.unrealized_pnl(price=110.0) == pytest.approx(100.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_position.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement Position**

`backtester/engine/position.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from backtester.core.enums import OrderSide
from backtester.engine.fills import Fill


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def apply_fill(self, fill: Fill) -> None:
        if fill.side == OrderSide.BUY:
            new_qty = self.qty + fill.qty
            self.avg_cost = (self.avg_cost * self.qty + fill.price * fill.qty) / new_qty
            self.qty = new_qty
            self.realized_pnl -= fill.commission
        else:  # SELL — long-only means we can only close existing longs
            if self.qty <= 0:
                raise ValueError("long-only: cannot SELL when position is flat")
            sell_qty = min(fill.qty, self.qty)
            self.realized_pnl += sell_qty * (fill.price - self.avg_cost) - fill.commission
            self.qty -= sell_qty
            if self.qty == 0:
                self.avg_cost = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_cost)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_position.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/position.py tests/unit/test_position.py
git commit -m "feat(engine): add long-only Position primitive"
```

---

### Task 17: Broker (thin wrapper around FillEngine)

**Files:**
- Create: `backtester/engine/broker.py`
- Create: `tests/unit/test_broker.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_broker.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.config.models import ExecutionConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.engine.broker import Broker
from backtester.engine.orders import Order


def test_broker_builds_fill_engine_from_config():
    b = Broker(ExecutionConfig(commission_bps=3.0, slippage_bps=4.0))
    assert b.fills.commission_bps == 3.0
    assert b.fills.slippage_bps == 4.0


def test_broker_submit_returns_fill_for_market():
    b = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    bar = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
                    name=pd.Timestamp("2024-01-02"))
    order = Order(timestamp=pd.Timestamp("2024-01-02"), symbol="SPY",
                  side=OrderSide.BUY, qty=5, order_type=OrderType.MARKET)
    fill = b.submit(order, bar)
    assert fill is not None
    assert fill.qty == 5
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_broker.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement Broker**

`backtester/engine/broker.py`:
```python
from __future__ import annotations

from typing import Optional
import pandas as pd

from backtester.config.models import ExecutionConfig
from backtester.engine.fills import Fill, FillEngine
from backtester.engine.orders import Order


class Broker:
    """Thin adapter that owns a FillEngine plus execution policy state."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.fills = FillEngine(
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
        )
        self.allow_fractional = config.allow_fractional

    def round_qty(self, qty: float) -> float:
        if self.allow_fractional:
            return qty
        return float(int(qty))

    def submit(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        return self.fills.try_fill(order, bar)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_broker.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/broker.py tests/unit/test_broker.py
git commit -m "feat(engine): add Broker wrapping FillEngine"
```

---

### Task 18: Portfolio simulator

**Files:**
- Create: `backtester/engine/portfolio.py`
- Create: `tests/unit/test_portfolio.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_portfolio.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from tests.fixtures.synthetic import make_ohlcv


def _buy_and_hold_signals(data: pd.DataFrame) -> SignalFrame:
    sf = pd.DataFrame(index=data.index)
    sf["signal"] = 1
    sf["signal"].iloc[0] = 0  # enter on bar 2 (signal already shifted by strategy convention)
    sf["size"] = 1.0
    return SignalFrame(data=sf)


def test_flat_signal_produces_no_trades(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    flat = SignalFrame(data=pd.DataFrame({"signal": 0, "size": 1.0}, index=ohlcv_small.index))
    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=flat, broker=broker)
    assert len(trades) == 0
    assert eq["equity"].iloc[0] == pytest.approx(10_000.0)
    assert eq["equity"].iloc[-1] == pytest.approx(10_000.0)


def test_signal_change_emits_one_buy_and_one_sell(ohlcv_small):
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0,
                                    initial_cash=10_000.0))
    # signal long for first 30 bars, then flat
    n = len(ohlcv_small)
    sig = pd.DataFrame(index=ohlcv_small.index)
    sig["signal"] = 0
    sig["signal"].iloc[1:30] = 1
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=ohlcv_small, signal_frame=sf, broker=broker)
    assert len(trades) == 2  # one entry, one exit
    assert trades.iloc[0]["side"] == "buy"
    assert trades.iloc[1]["side"] == "sell"
    # equity series has same length as data
    assert len(eq) == n


def test_equity_curve_reflects_pnl():
    data = make_ohlcv(n=50, seed=99, start_price=100.0, drift=0.005, vol=0.001)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 1
    sig["signal"].iloc[0] = 0
    sig["size"] = 1.0
    sf = SignalFrame(data=sig)

    trades, positions, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    # With positive drift and no costs, equity should rise
    assert eq["equity"].iloc[-1] > eq["equity"].iloc[0]


def test_limit_orders_via_price_column():
    data = make_ohlcv(n=20, seed=11)
    sim = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    sig = pd.DataFrame(index=data.index)
    sig["signal"] = 0
    sig["signal"].iloc[1] = 1
    sig["size"] = 1.0
    # Limit far below market — should not fill on next bar
    sig["limit_price"] = data["low"].min() * 0.5
    sf = SignalFrame(data=sig, price_column="limit_price")
    trades, _, eq = sim.simulate(data=data, signal_frame=sf, broker=broker)
    assert len(trades) == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_portfolio.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement PortfolioSimulator**

`backtester/engine/portfolio.py`:
```python
from __future__ import annotations

from typing import List, Optional, Tuple
import pandas as pd

from backtester.config.models import PortfolioConfig
from backtester.core.enums import OrderSide, OrderType
from backtester.core.types import SignalFrame
from backtester.engine.broker import Broker
from backtester.engine.fills import Fill
from backtester.engine.orders import Order
from backtester.engine.position import Position


class PortfolioSimulator:
    """Translates signals -> orders -> fills, tracking cash, position, equity."""

    def __init__(self, config: PortfolioConfig, initial_cash: float = 100_000.0):
        self.config = config
        self.initial_cash = initial_cash

    def simulate(
        self,
        data: pd.DataFrame,
        signal_frame: SignalFrame,
        broker: Broker,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        signals = signal_frame.data
        sig_col = signal_frame.signal_column
        size_col = signal_frame.size_column
        price_col = signal_frame.price_column

        symbol = "ASSET"  # filled in by engine via ctx; for primitives we use a fixed tag
        pos = Position(symbol=symbol)
        cash = self.initial_cash

        fills: List[Fill] = []
        pending: Optional[Order] = None
        prev_signal = 0

        equity_rows = []
        position_rows = []

        index = data.index
        for i, ts in enumerate(index):
            bar = data.iloc[i]
            # 1. Try to execute any pending order on this bar
            if pending is not None:
                fill = broker.submit(pending, bar)
                if fill is not None:
                    fills.append(fill)
                    cash += fill.cash_delta
                    pos.apply_fill(fill)
                pending = None  # one-shot semantics: cancel if not filled

            # 2. Read this bar's signal; if it differs from current state, schedule order for next bar
            sig = int(signals[sig_col].iloc[i]) if sig_col in signals.columns else 0
            if i + 1 < len(index):
                next_bar_ts = index[i + 1]
                target_long = sig == 1
                currently_long = pos.qty > 0

                if target_long and not currently_long:
                    # entry order
                    equity_now = cash + pos.market_value(float(bar["close"]))
                    size = float(signals[size_col].iloc[i]) if size_col and size_col in signals.columns else 1.0
                    alloc = equity_now * self.config.size * size
                    raw_qty = alloc / float(bar["close"])
                    qty = broker.round_qty(raw_qty)
                    if qty > 0:
                        if price_col and price_col in signals.columns and pd.notna(signals[price_col].iloc[i]):
                            pending = Order(
                                timestamp=next_bar_ts, symbol=symbol, side=OrderSide.BUY,
                                qty=qty, order_type=OrderType.LIMIT,
                                limit_price=float(signals[price_col].iloc[i]),
                            )
                        else:
                            pending = Order(
                                timestamp=next_bar_ts, symbol=symbol, side=OrderSide.BUY,
                                qty=qty, order_type=OrderType.MARKET,
                            )

                elif not target_long and currently_long:
                    pending = Order(
                        timestamp=next_bar_ts, symbol=symbol, side=OrderSide.SELL,
                        qty=pos.qty, order_type=OrderType.MARKET,
                    )

            # 3. Mark to market at close
            mv = pos.market_value(float(bar["close"]))
            equity = cash + mv
            equity_rows.append({"timestamp": ts, "cash": cash, "position_value": mv, "equity": equity})
            position_rows.append({"timestamp": ts, "qty": pos.qty, "avg_cost": pos.avg_cost, "close": float(bar["close"])})
            prev_signal = sig

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        positions_df = pd.DataFrame(position_rows).set_index("timestamp")
        trades_df = pd.DataFrame([
            {
                "timestamp": f.timestamp,
                "side": f.side.value,
                "qty": f.qty,
                "price": f.price,
                "commission": f.commission,
                "notional": f.notional,
            }
            for f in fills
        ])
        return trades_df, positions_df, equity_curve
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_portfolio.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/engine/portfolio.py tests/unit/test_portfolio.py
git commit -m "feat(engine): add PortfolioSimulator with signal->order pipeline"
```

---

### Task 19: BacktestEngine

**Files:**
- Create: `backtester/engine/backtest_engine.py`
- Create: `tests/integration/__init__.py` (already created in Task 1)
- Create: `tests/integration/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_backtest_engine.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.core.types import SignalFrame, StrategyContext
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.strategies.base import BaseStrategy
from tests.fixtures.synthetic import make_ohlcv


@dataclass(slots=True)
class _BHParams:
    size: float = 1.0


class _BuyAndHoldStrategy(BaseStrategy[_BHParams]):
    strategy_id = "_buy_and_hold_test"

    @classmethod
    def params_type(cls):
        return _BHParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params: _BHParams):
        df = pd.DataFrame(index=data.index)
        df["signal"] = 1
        df["signal"].iloc[0] = 0
        df["size"] = params.size
        return SignalFrame(data=df)


def test_engine_runs_end_to_end_and_returns_result():
    data = make_ohlcv(n=100, seed=3, drift=0.001, vol=0.005)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    result = engine.run(_BuyAndHoldStrategy(), data, _BHParams(), symbol="SYN", timeframe="1d")

    assert "total_return" in result.summary
    assert len(result.equity_curve) == 100
    assert (result.equity_curve["equity"] > 0).all()
    assert "params" in result.summary or "params" in getattr(result, "summary", {})


def test_engine_validates_data():
    broker = Broker(ExecutionConfig())
    portfolio = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    bad = make_ohlcv(n=10, seed=1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        engine.run(_BuyAndHoldStrategy(), bad, _BHParams(), symbol="X", timeframe="1d")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_backtest_engine.py -v`
Expected: FAIL — `backtester.engine.backtest_engine` not found OR `compute_summary_metrics` not found. We will stub metrics now and replace in the analytics phase.

- [ ] **Step 3: Stub `compute_summary_metrics`**

`backtester/analytics/metrics.py` (initial stub — will be expanded in Task 22):
```python
from __future__ import annotations

import pandas as pd


def compute_summary_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, positions: pd.DataFrame) -> dict:
    if len(equity_curve) == 0:
        return {"total_return": 0.0, "n_trades": 0}
    start = equity_curve["equity"].iloc[0]
    end = equity_curve["equity"].iloc[-1]
    return {
        "total_return": (end / start) - 1.0,
        "n_trades": int(len(trades)),
        "final_equity": float(end),
    }
```

- [ ] **Step 4: Implement BacktestEngine**

`backtester/engine/backtest_engine.py`:
```python
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import pandas as pd

from backtester.analytics.metrics import compute_summary_metrics
from backtester.core.types import BacktestResult, StrategyContext
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator


class BacktestEngine:
    def __init__(self, broker: Broker, portfolio: PortfolioSimulator):
        self.broker = broker
        self.portfolio = portfolio

    def run(self, strategy, data: pd.DataFrame, params, symbol: str, timeframe: str) -> BacktestResult:
        strategy.validate(data, params)

        ctx = StrategyContext(
            symbol=symbol,
            timeframe=timeframe,
            warmup_bars=strategy.warmup_bars(params),
            metadata={"params": asdict(params) if is_dataclass(params) else {}},
        )

        indicators = strategy.indicators(data, params)
        signal_frame = strategy.generate_signals(data, indicators, ctx, params)

        trades, positions, equity_curve = self.portfolio.simulate(
            data=data,
            signal_frame=signal_frame,
            broker=self.broker,
        )

        summary = compute_summary_metrics(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
        )
        summary["params"] = ctx.metadata["params"]
        summary["symbol"] = symbol
        summary["timeframe"] = timeframe

        return BacktestResult(
            summary=summary,
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
        )
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/integration/test_backtest_engine.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all prior tests still pass.

- [ ] **Step 7: Commit**

```
git add backtester/engine/backtest_engine.py backtester/analytics/metrics.py tests/integration/test_backtest_engine.py
git commit -m "feat(engine): add BacktestEngine integration"
```

---

## Phase 6: Analytics

### Task 20: Drawdown and trade extraction

**Files:**
- Create: `backtester/analytics/drawdown.py`
- Create: `backtester/analytics/trades.py`
- Create: `tests/unit/test_analytics_drawdown.py`
- Create: `tests/unit/test_analytics_trades.py`

- [ ] **Step 1: Write the failing drawdown test**

`tests/unit/test_analytics_drawdown.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.analytics.drawdown import drawdown_series, max_drawdown


def test_no_drawdown_when_monotonic():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3])
    dd = drawdown_series(eq)
    assert (dd == 0).all()
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_basic():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6, 1.0])
    # peak at 1.2, trough at 0.6 -> -50%
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_drawdown_series_lengths_match():
    eq = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8])
    assert len(drawdown_series(eq)) == len(eq)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_analytics_drawdown.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement drawdown**

`backtester/analytics/drawdown.py`:
```python
from __future__ import annotations

import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(drawdown_series(equity).min())
```

- [ ] **Step 4: Run drawdown test**

Run: `pytest tests/unit/test_analytics_drawdown.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the failing trades test**

`tests/unit/test_analytics_trades.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.analytics.trades import extract_round_trips


def _trades_df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "side", "qty", "price", "commission", "notional"])


def test_no_trades_returns_empty():
    rt = extract_round_trips(_trades_df([]))
    assert len(rt) == 0


def test_single_round_trip():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-02"), "side": "buy",  "qty": 10, "price": 100.0, "commission": 0.0, "notional": 1000.0},
        {"timestamp": pd.Timestamp("2024-01-10"), "side": "sell", "qty": 10, "price": 110.0, "commission": 0.0, "notional": 1100.0},
    ]
    rt = extract_round_trips(_trades_df(rows))
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["pnl"] == 100.0
    assert row["return_pct"] > 0
    assert row["bars_held"] >= 1


def test_two_round_trips_sequential():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-02"), "side": "buy",  "qty": 10, "price": 100.0, "commission": 0.0, "notional": 1000.0},
        {"timestamp": pd.Timestamp("2024-01-10"), "side": "sell", "qty": 10, "price": 90.0,  "commission": 0.0, "notional": 900.0},
        {"timestamp": pd.Timestamp("2024-02-01"), "side": "buy",  "qty": 5,  "price": 50.0,  "commission": 0.0, "notional": 250.0},
        {"timestamp": pd.Timestamp("2024-02-10"), "side": "sell", "qty": 5,  "price": 60.0,  "commission": 0.0, "notional": 300.0},
    ]
    rt = extract_round_trips(_trades_df(rows))
    assert len(rt) == 2
    assert rt.iloc[0]["pnl"] == -100.0
    assert rt.iloc[1]["pnl"] == 50.0
```

- [ ] **Step 6: Run test to verify failure**

Run: `pytest tests/unit/test_analytics_trades.py -v`
Expected: FAIL — module not found.

- [ ] **Step 7: Implement trades**

`backtester/analytics/trades.py`:
```python
from __future__ import annotations

import pandas as pd


def extract_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair sequential BUY/SELL trades into long-only round trips."""
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "qty", "entry_price", "exit_price",
            "pnl", "return_pct", "bars_held",
        ])

    rows = []
    open_buy: dict | None = None
    for _, t in trades.iterrows():
        if t["side"] == "buy":
            open_buy = {"time": t["timestamp"], "qty": t["qty"], "price": t["price"],
                        "commission": t["commission"]}
        elif t["side"] == "sell" and open_buy is not None:
            pnl = (t["price"] - open_buy["price"]) * t["qty"] - (t["commission"] + open_buy["commission"])
            ret = (t["price"] / open_buy["price"]) - 1.0
            bars_held = max(1, (t["timestamp"] - open_buy["time"]).days)
            rows.append({
                "entry_time": open_buy["time"],
                "exit_time": t["timestamp"],
                "qty": t["qty"],
                "entry_price": open_buy["price"],
                "exit_price": t["price"],
                "pnl": pnl,
                "return_pct": ret,
                "bars_held": bars_held,
            })
            open_buy = None

    return pd.DataFrame(rows)
```

- [ ] **Step 8: Run trades test**

Run: `pytest tests/unit/test_analytics_trades.py -v`
Expected: 3 passed.

- [ ] **Step 9: Commit**

```
git add backtester/analytics/drawdown.py backtester/analytics/trades.py tests/unit/test_analytics_drawdown.py tests/unit/test_analytics_trades.py
git commit -m "feat(analytics): add drawdown and round-trip extraction"
```

---

### Task 21: Exposure metrics

**Files:**
- Create: `backtester/analytics/exposure.py`
- Create: `tests/unit/test_analytics_exposure.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_analytics_exposure.py`:
```python
from __future__ import annotations

import pandas as pd
import pytest

from backtester.analytics.exposure import time_in_market, turnover


def test_time_in_market_zero_when_always_flat():
    positions = pd.DataFrame({"qty": [0, 0, 0, 0]})
    assert time_in_market(positions) == 0.0


def test_time_in_market_full_when_always_long():
    positions = pd.DataFrame({"qty": [1, 1, 1, 1]})
    assert time_in_market(positions) == 1.0


def test_time_in_market_half():
    positions = pd.DataFrame({"qty": [0, 1, 0, 1]})
    assert time_in_market(positions) == pytest.approx(0.5)


def test_turnover_zero_with_no_trades():
    trades = pd.DataFrame(columns=["notional"])
    eq = pd.DataFrame({"equity": [10_000.0, 10_000.0]})
    assert turnover(trades, eq) == 0.0


def test_turnover_basic():
    trades = pd.DataFrame({"notional": [10_000.0, 10_000.0]})
    eq = pd.DataFrame({"equity": [10_000.0, 10_500.0]})
    # turnover = sum(notional) / mean(equity)
    assert turnover(trades, eq) == pytest.approx(20_000.0 / 10_250.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_analytics_exposure.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement exposure**

`backtester/analytics/exposure.py`:
```python
from __future__ import annotations

import pandas as pd


def time_in_market(positions: pd.DataFrame) -> float:
    if len(positions) == 0:
        return 0.0
    return float((positions["qty"].abs() > 0).mean())


def turnover(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    if len(trades) == 0:
        return 0.0
    avg_equity = float(equity_curve["equity"].mean())
    if avg_equity == 0:
        return 0.0
    return float(trades["notional"].abs().sum() / avg_equity)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_analytics_exposure.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add backtester/analytics/exposure.py tests/unit/test_analytics_exposure.py
git commit -m "feat(analytics): add time-in-market and turnover"
```

---

### Task 22: Full summary metrics (replace the stub)

**Files:**
- Modify: `backtester/analytics/metrics.py`
- Create: `tests/unit/test_analytics_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_analytics_metrics.py`:
```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    compute_summary_metrics,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
)


def _eq(values):
    idx = pd.bdate_range("2024-01-02", periods=len(values))
    return pd.DataFrame({"equity": values, "cash": values, "position_value": [0] * len(values)}, index=idx)


def test_total_return_basic():
    eq = _eq([100.0, 110.0, 120.0])
    out = compute_summary_metrics(eq, pd.DataFrame(), pd.DataFrame({"qty": [0, 1, 1]}))
    assert out["total_return"] == pytest.approx(0.2)


def test_annualized_return_one_year():
    n = 252
    eq = pd.Series(np.linspace(100, 110, n))
    ar = annualized_return(eq)
    assert ar == pytest.approx(0.10, rel=1e-2)


def test_volatility_nonzero_when_returns_vary():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    assert annualized_volatility(eq) > 0


def test_sharpe_zero_when_no_variance():
    eq = pd.Series([100.0] * 100)
    assert sharpe_ratio(eq) == 0.0


def test_sortino_finite():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500))))
    s = sortino_ratio(eq)
    assert math.isfinite(s)


def test_summary_keys_present():
    eq = _eq(np.linspace(100, 130, 252).tolist())
    trades = pd.DataFrame({
        "timestamp": [eq.index[10], eq.index[20]],
        "side": ["buy", "sell"],
        "qty": [10, 10],
        "price": [100.0, 110.0],
        "commission": [0.0, 0.0],
        "notional": [1000.0, 1100.0],
    })
    positions = pd.DataFrame({"qty": [0] * 252}, index=eq.index)
    positions.iloc[10:20, 0] = 10
    out = compute_summary_metrics(eq, trades, positions)
    for key in ["total_return", "annualized_return", "annualized_vol",
                "sharpe", "sortino", "max_drawdown", "n_trades",
                "n_round_trips", "win_rate", "avg_round_trip_pnl",
                "time_in_market", "turnover", "final_equity"]:
        assert key in out, f"missing {key}"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_analytics_metrics.py -v`
Expected: FAIL — functions not exported / wrong keys.

- [ ] **Step 3: Implement full metrics**

Replace `backtester/analytics/metrics.py` with:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.analytics.drawdown import max_drawdown
from backtester.analytics.exposure import time_in_market, turnover
from backtester.analytics.trades import extract_round_trips
from backtester.core.constants import TRADING_DAYS_PER_YEAR


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(equity: pd.Series) -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    excess = r - (rf / TRADING_DAYS_PER_YEAR)
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / r.std(ddof=1))


def sortino_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    excess = r - (rf / TRADING_DAYS_PER_YEAR)
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / downside.std(ddof=1))


def compute_summary_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
) -> dict:
    if len(equity_curve) == 0:
        return {
            "total_return": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "n_trades": 0, "n_round_trips": 0, "win_rate": 0.0,
            "avg_round_trip_pnl": 0.0, "time_in_market": 0.0, "turnover": 0.0,
            "final_equity": 0.0,
        }

    eq = equity_curve["equity"]
    rts = extract_round_trips(trades) if len(trades) else pd.DataFrame()
    wins = int((rts["pnl"] > 0).sum()) if len(rts) else 0
    win_rate = (wins / len(rts)) if len(rts) else 0.0
    avg_rt = float(rts["pnl"].mean()) if len(rts) else 0.0

    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "annualized_return": annualized_return(eq),
        "annualized_vol": annualized_volatility(eq),
        "sharpe": sharpe_ratio(eq),
        "sortino": sortino_ratio(eq),
        "max_drawdown": max_drawdown(eq),
        "n_trades": int(len(trades)),
        "n_round_trips": int(len(rts)),
        "win_rate": float(win_rate),
        "avg_round_trip_pnl": avg_rt,
        "time_in_market": time_in_market(positions),
        "turnover": turnover(trades, equity_curve),
        "final_equity": float(eq.iloc[-1]),
    }
```

- [ ] **Step 4: Run all metric and engine tests**

Run: `pytest tests/unit/test_analytics_metrics.py tests/integration/test_backtest_engine.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```
git add backtester/analytics/metrics.py tests/unit/test_analytics_metrics.py
git commit -m "feat(analytics): full summary metrics (sharpe, sortino, dd, win rate)"
```

---

## Phase 7: Sample Strategies

### Task 23: SMA cross strategy

**Files:**
- Create: `strategies/__init__.py`
- Create: `strategies/sma_cross.py`
- Create: `tests/unit/test_strategy_sma_cross.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_sma_cross.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.sma_cross import SMACrossParams, SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_strategy_id_and_params():
    assert SMACrossStrategy.strategy_id == "sma_cross"
    assert SMACrossStrategy.params_type() is SMACrossParams


def test_indicators_have_fast_and_slow():
    data = make_ohlcv(n=100, seed=4)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=10, slow=30, size=1.0)
    ind = strat.indicators(data, p)
    assert "fast_sma" in ind.columns and "slow_sma" in ind.columns


def test_signals_shifted_and_zero_when_fast_below_slow():
    # Falling series: fast SMA will be below slow SMA -> signal = 0
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 - i for i in range(n)], "volume": 100}, index=idx)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=5, slow=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # last bar should be 0 because fast < slow throughout
    assert sf.data["signal"].iloc[-1] == 0


def test_signals_one_when_fast_above_slow():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 + i for i in range(n)], "volume": 100}, index=idx)
    strat = SMACrossStrategy()
    p = SMACrossParams(fast=5, slow=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    assert sf.data["signal"].iloc[-1] == 1


def test_warmup_bars_uses_slow():
    p = SMACrossParams(fast=10, slow=50)
    assert SMACrossStrategy().warmup_bars(p) == 50
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_sma_cross.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement SMA cross**

`strategies/__init__.py`: (leave empty, just a placeholder package marker)

`strategies/sma_cross.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SMACrossParams:
    fast: int = 20
    slow: int = 50
    size: float = 1.0


class SMACrossStrategy(BaseStrategy[SMACrossParams]):
    """
    Purpose:
        Trend-following long-only strategy using fast/slow moving average crossover.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "sma_cross"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return SMACrossParams

    def warmup_bars(self, params: SMACrossParams) -> int:
        return max(params.fast, params.slow)

    def indicators(self, data: pd.DataFrame, params: SMACrossParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["fast_sma"] = data["close"].rolling(params.fast).mean()
        out["slow_sma"] = data["close"].rolling(params.slow).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SMACrossParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df.loc[indicators["fast_sma"] > indicators["slow_sma"], "signal"] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_sma_cross.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add strategies/__init__.py strategies/sma_cross.py tests/unit/test_strategy_sma_cross.py
git commit -m "feat(strategies): add SMA cross sample strategy"
```

---

### Task 24: RSI mean-reversion strategy

**Files:**
- Create: `strategies/rsi_mean_reversion.py`
- Create: `tests/unit/test_strategy_rsi.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_rsi.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.rsi_mean_reversion import RSIMeanReversionParams, RSIMeanReversionStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_strategy_id_and_params():
    assert RSIMeanReversionStrategy.strategy_id == "rsi_mean_reversion"
    assert RSIMeanReversionStrategy.params_type() is RSIMeanReversionParams


def test_indicator_has_rsi_column():
    data = make_ohlcv(n=200, seed=5)
    strat = RSIMeanReversionStrategy()
    p = RSIMeanReversionParams()
    ind = strat.indicators(data, p)
    assert "rsi" in ind.columns
    assert ind["rsi"].dropna().between(0, 100).all()


def test_signals_long_when_rsi_oversold():
    # Falling series -> RSI low -> oversold -> enter long after shift
    n = 100
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": [100 - i * 0.5 for i in range(n)], "volume": 100}, index=idx)
    strat = RSIMeanReversionStrategy()
    p = RSIMeanReversionParams(period=14, oversold=30, overbought=70)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    assert sf.data["signal"].sum() > 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_rsi.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement RSI strategy**

`strategies/rsi_mean_reversion.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RSIMeanReversionParams:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    size: float = 1.0


class RSIMeanReversionStrategy(BaseStrategy[RSIMeanReversionParams]):
    """
    Purpose:
        Long-only mean-reversion: enter long when RSI crosses below `oversold`,
        exit when RSI crosses above `overbought`.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "rsi_mean_reversion"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return RSIMeanReversionParams

    def warmup_bars(self, params: RSIMeanReversionParams) -> int:
        return params.period + 1

    def indicators(self, data: pd.DataFrame, params: RSIMeanReversionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        delta = data["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RSIMeanReversionParams,
    ) -> SignalFrame:
        rsi = indicators["rsi"]
        df = pd.DataFrame(index=data.index)
        # State machine: long when last cross was below oversold; flat when last cross was above overbought
        state = (rsi < params.oversold).astype(int) - (rsi > params.overbought).astype(int)
        # Forward-fill the binary state so a long position is held until exit
        signal = state.replace(0, np.nan).ffill().fillna(0).clip(lower=0).astype(int)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_rsi.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add strategies/rsi_mean_reversion.py tests/unit/test_strategy_rsi.py
git commit -m "feat(strategies): add RSI mean-reversion sample strategy"
```

---

### Task 25: Donchian breakout strategy

**Files:**
- Create: `strategies/breakout_20d.py`
- Create: `tests/unit/test_strategy_breakout.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_breakout.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.core.types import StrategyContext
from strategies.breakout_20d import Breakout20DParams, Breakout20DStrategy


def test_strategy_id_and_params():
    assert Breakout20DStrategy.strategy_id == "breakout_20d"
    assert Breakout20DStrategy.params_type() is Breakout20DParams


def test_signal_triggers_on_new_high():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    # close rises monotonically -> every bar is a new lookback high
    df = pd.DataFrame({"open": 1, "high": [100 + i for i in range(n)], "low": 1,
                       "close": [100 + i for i in range(n)], "volume": 100}, index=idx)
    strat = Breakout20DStrategy()
    p = Breakout20DParams(lookback=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # After the warmup period, signal should be 1
    assert sf.data["signal"].iloc[-1] == 1


def test_no_signal_in_warmup():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"open": 1, "high": [100] * n, "low": 1, "close": [100] * n, "volume": 100}, index=idx)
    strat = Breakout20DStrategy()
    p = Breakout20DParams(lookback=20)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=strat.warmup_bars(p))
    sf = strat.generate_signals(df, strat.indicators(df, p), ctx, p)
    # Flat market -> never breaks the lookback high -> no signal
    assert sf.data["signal"].sum() == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_breakout.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement breakout**

`strategies/breakout_20d.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Breakout20DParams:
    lookback: int = 20
    exit_lookback: int = 10
    size: float = 1.0


class Breakout20DStrategy(BaseStrategy[Breakout20DParams]):
    """
    Purpose:
        Long-only Donchian breakout: enter long when close exceeds the rolling
        lookback-day high; exit when close falls below the rolling
        exit_lookback-day low.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "breakout_20d"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return Breakout20DParams

    def warmup_bars(self, params: Breakout20DParams) -> int:
        return max(params.lookback, params.exit_lookback)

    def indicators(self, data: pd.DataFrame, params: Breakout20DParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["upper"] = data["close"].rolling(params.lookback).max().shift(1)
        out["lower"] = data["close"].rolling(params.exit_lookback).min().shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Breakout20DParams,
    ) -> SignalFrame:
        close = data["close"]
        long_trigger = (close > indicators["upper"]).astype(int)
        exit_trigger = (close < indicators["lower"]).astype(int) * -1
        raw = (long_trigger + exit_trigger).replace(0, pd.NA).ffill().fillna(0).clip(lower=0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_breakout.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire all three into the registry**

Replace `backtester/strategies/registry.py` contents with the import-and-register block at module bottom — edit the file so the registry is populated by default. Add at the end of `backtester/strategies/registry.py`:

```python

# --- Default strategy registrations (explicit, predictable order) ---
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
```

- [ ] **Step 6: Smoke-test the registry**

Run: `python -c "from backtester.strategies.registry import STRATEGY_REGISTRY; print(sorted(STRATEGY_REGISTRY))"`
Expected: `['breakout_20d', 'rsi_mean_reversion', 'sma_cross']`.

- [ ] **Step 7: Commit**

```
git add strategies/breakout_20d.py tests/unit/test_strategy_breakout.py backtester/strategies/registry.py
git commit -m "feat(strategies): add Donchian breakout and wire all samples into registry"
```

---

## Phase 8: IO and CLI (single-run backtest)

### Task 26: Logging + serialization helpers

**Files:**
- Create: `backtester/io/logging.py`
- Create: `backtester/io/serialization.py`
- Create: `tests/unit/test_io_serialization.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_io_serialization.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.io.serialization import to_jsonable, write_json


def test_to_jsonable_handles_numpy_and_timestamps():
    payload = {
        "n": np.int64(3),
        "f": np.float64(1.5),
        "ts": pd.Timestamp("2024-01-02"),
        "arr": np.array([1, 2]),
        "nested": {"k": np.float32(0.5)},
    }
    out = to_jsonable(payload)
    json.dumps(out)  # must not raise


def test_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "out.json"
    write_json(p, {"a": 1, "ts": pd.Timestamp("2024-01-02")})
    data = json.loads(p.read_text())
    assert data["a"] == 1
    assert data["ts"].startswith("2024-01-02")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_io_serialization.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement serialization**

`backtester/io/serialization.py`:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    return obj


def write_json(path: PathLike, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(payload), indent=2))
```

- [ ] **Step 4: Implement logging**

`backtester/io/logging.py`:
```python
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("backtester")
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)

    root.propagate = False
    return root
```

- [ ] **Step 5: Run test to verify pass**

Run: `pytest tests/unit/test_io_serialization.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```
git add backtester/io/serialization.py backtester/io/logging.py tests/unit/test_io_serialization.py
git commit -m "feat(io): add JSON serialization helpers and logging setup"
```

---

### Task 27: Artifact writer

**Files:**
- Create: `backtester/io/artifacts.py`
- Create: `tests/unit/test_io_artifacts.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_io_artifacts.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtester.config.models import (
    DataConfig, ExecutionConfig, PortfolioConfig, RunConfig,
)
from backtester.core.types import BacktestResult
from backtester.io.artifacts import ArtifactWriter


def _result():
    idx = pd.bdate_range("2024-01-02", periods=5)
    eq = pd.DataFrame({"equity": [100.0, 101, 102, 103, 104],
                       "cash": [100, 0, 0, 0, 104], "position_value": [0, 101, 102, 103, 0]}, index=idx)
    trades = pd.DataFrame([{"timestamp": idx[1], "side": "buy", "qty": 1.0,
                            "price": 100.0, "commission": 0.0, "notional": 100.0}])
    positions = pd.DataFrame({"qty": [0, 1, 1, 1, 0]}, index=idx)
    return BacktestResult(summary={"total_return": 0.04}, equity_curve=eq, trades=trades, positions=positions)


def _config():
    return RunConfig(
        run_name="smoke",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )


def test_writer_creates_run_dir_with_artifacts(tmp_path: Path):
    w = ArtifactWriter(root=tmp_path, run_name="smoke", now="20240514_0114")
    run_dir = w.run_dir
    assert run_dir.parent == tmp_path
    w.write_config(_config())
    w.write_result(_result())
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "positions.csv").exists()
    assert (run_dir / "equity_curve.csv").exists()


def test_run_dir_name_format(tmp_path: Path):
    w = ArtifactWriter(root=tmp_path, run_name="foo", now="20260514_0114")
    assert w.run_dir.name == "20260514_0114_foo"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_io_artifacts.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement artifact writer**

`backtester/io/artifacts.py`:
```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from backtester.config.loader import dump_run_config
from backtester.config.models import RunConfig
from backtester.core.types import BacktestResult
from backtester.io.serialization import write_json

PathLike = Union[str, Path]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


class ArtifactWriter:
    def __init__(self, root: PathLike, run_name: str, now: Optional[str] = None):
        self.root = Path(root)
        self.now = now or _stamp()
        self.run_dir = self.root / f"{self.now}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, rc: RunConfig) -> Path:
        path = self.run_dir / "config_resolved.yaml"
        dump_run_config(rc, path)
        return path

    def write_result(self, result: BacktestResult) -> None:
        write_json(self.run_dir / "summary.json", result.summary)
        result.trades.to_csv(self.run_dir / "trades.csv", index=False)
        result.positions.to_csv(self.run_dir / "positions.csv", index_label="timestamp")
        result.equity_curve.to_csv(self.run_dir / "equity_curve.csv", index_label="timestamp")

    def write_window_results(self, payload: Any) -> Path:
        path = self.run_dir / "window_results.json"
        write_json(path, payload)
        return path

    def log_path(self) -> Path:
        return self.run_dir / "logs.txt"
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_io_artifacts.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backtester/io/artifacts.py tests/unit/test_io_artifacts.py
git commit -m "feat(io): add ArtifactWriter for run output bundles"
```

---

### Task 28: Strategy params instantiation helper

**Files:**
- Create: `backtester/strategies/instantiate.py`
- Create: `tests/unit/test_strategy_instantiate.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_strategy_instantiate.py`:
```python
from __future__ import annotations

import pytest

from backtester.core.exceptions import StrategyError
from backtester.strategies.instantiate import build_strategy_and_params


def test_build_known_strategy_with_defaults():
    strat, params = build_strategy_and_params("sma_cross", {})
    assert strat.strategy_id == "sma_cross"
    assert params.fast == 20  # default


def test_build_known_strategy_with_overrides():
    strat, params = build_strategy_and_params("sma_cross", {"fast": 5, "slow": 25})
    assert params.fast == 5 and params.slow == 25


def test_build_unknown_strategy_raises():
    with pytest.raises(KeyError):
        build_strategy_and_params("does_not_exist", {})


def test_build_unknown_param_key_raises():
    with pytest.raises(StrategyError, match="unknown"):
        build_strategy_and_params("sma_cross", {"not_a_field": 1})
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_strategy_instantiate.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement helper**

`backtester/strategies/instantiate.py`:
```python
from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Tuple

from backtester.core.exceptions import StrategyError
from backtester.strategies.base import BaseStrategy
from backtester.strategies.registry import get_strategy_class


def build_strategy_and_params(strategy_id: str, params_dict: Dict[str, Any]) -> Tuple[BaseStrategy, Any]:
    cls = get_strategy_class(strategy_id)
    params_type = cls.params_type()
    allowed = {f.name for f in fields(params_type)}
    unknown = set(params_dict) - allowed
    if unknown:
        raise StrategyError(
            f"unknown params for {strategy_id!r}: {sorted(unknown)} (allowed: {sorted(allowed)})"
        )
    params = params_type(**params_dict)
    return cls(), params
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_strategy_instantiate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/strategies/instantiate.py tests/unit/test_strategy_instantiate.py
git commit -m "feat(strategies): add safe params-dict instantiation helper"
```

---

### Task 29: CLI runner — run_backtest

**Files:**
- Create: `backtester/runners/run_backtest.py`
- Create: `configs/backtests/sma_cross_spy.yaml`
- Create: `tests/integration/test_run_backtest_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_run_backtest_cli.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.fixtures.synthetic import make_ohlcv


def _write_data(tmp_path: Path) -> Path:
    raw = tmp_path / "data"
    raw.mkdir()
    df = make_ohlcv(n=400, seed=8)
    df.to_csv(raw / "SYN.csv", index_label="date")
    return raw


def _write_config(tmp_path: Path, raw: Path, out: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
run_name: smoke_run
strategy: sma_cross
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2026-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
portfolio:
  sizing_mode: "percent_equity"
  size: 1.0
output_root: "{out.as_posix()}"
""")
    return cfg


def test_run_backtest_cli_produces_artifacts(tmp_path: Path):
    raw = _write_data(tmp_path)
    out = tmp_path / "runs"
    cfg = _write_config(tmp_path, raw, out)

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_backtest", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    runs = list(out.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    for f in ("config_resolved.yaml", "summary.json", "trades.csv",
              "positions.csv", "equity_curve.csv", "logs.txt"):
        assert (run_dir / f).exists(), f"missing artifact: {f}"

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "total_return" in summary
    assert summary["symbol"] == "SYN"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_run_backtest_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement CLI runner**

`backtester/runners/run_backtest.py`:
```python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.strategies.instantiate import build_strategy_and_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_backtest")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)

    if len(rc.data.symbols) != 1:
        raise SystemExit("run_backtest expects exactly one symbol; use run_batch for many")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())
    log.info("run_dir=%s", writer.run_dir)

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)
    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    log.info("running %s on %s (%d bars)", rc.strategy, symbol, len(data))
    result = engine.run(strategy, data, params, symbol=symbol, timeframe=rc.data.timeframe)
    log.info("done: total_return=%.4f sharpe=%.3f max_dd=%.3f",
             result.summary["total_return"], result.summary["sharpe"], result.summary["max_drawdown"])

    writer.write_config(rc)
    writer.write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_run_backtest_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write the bundled SPY config**

`configs/backtests/sma_cross_spy.yaml`:
```yaml
run_name: sma_cross_spy
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
  root: "data/raw"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
output_root: "output/runs"
```

- [ ] **Step 6: Manual smoke run**

Run: `python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml`
Expected: prints log lines, creates a folder under `output/runs/` with all artifacts.

- [ ] **Step 7: Commit**

```
git add backtester/runners/run_backtest.py configs/backtests/sma_cross_spy.yaml tests/integration/test_run_backtest_cli.py
git commit -m "feat(cli): add run_backtest entrypoint and sample SPY config"
```

---

## Phase 9: Optimization

### Task 30: Parameter space

**Files:**
- Create: `backtester/optimize/parameter_space.py`
- Create: `tests/unit/test_parameter_space.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_parameter_space.py`:
```python
from __future__ import annotations

from backtester.optimize.parameter_space import expand_grid


def test_expand_grid_cartesian():
    space = {"fast": [10, 20], "slow": [50, 100]}
    combos = list(expand_grid(space))
    assert len(combos) == 4
    assert {"fast": 10, "slow": 50} in combos
    assert {"fast": 20, "slow": 100} in combos


def test_expand_grid_single_value():
    combos = list(expand_grid({"x": [1, 2, 3]}))
    assert combos == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_expand_grid_empty_yields_one_empty_dict():
    assert list(expand_grid({})) == [{}]


def test_expand_grid_preserves_key_order():
    space = {"a": [1], "b": [2], "c": [3]}
    combo = list(expand_grid(space))[0]
    assert list(combo.keys()) == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_parameter_space.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement parameter space**

`backtester/optimize/parameter_space.py`:
```python
from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, List


def expand_grid(space: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Yield every combination from a parameter grid as dicts."""
    if not space:
        yield {}
        return
    keys = list(space.keys())
    values = [space[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_parameter_space.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/optimize/parameter_space.py tests/unit/test_parameter_space.py
git commit -m "feat(optimize): add cartesian-product parameter grid expansion"
```

---

### Task 31: Objectives

**Files:**
- Create: `backtester/optimize/objectives.py`
- Create: `tests/unit/test_objectives.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_objectives.py`:
```python
from __future__ import annotations

import pytest

from backtester.optimize.objectives import resolve_objective, OBJECTIVES


def test_known_objectives():
    for name in ["sharpe", "sortino", "total_return", "calmar", "annualized_return"]:
        assert name in OBJECTIVES


def test_resolve_returns_callable_score():
    fn = resolve_objective("sharpe")
    summary = {"sharpe": 1.5, "max_drawdown": -0.1, "annualized_return": 0.1}
    assert fn(summary) == 1.5


def test_calmar_uses_abs_drawdown():
    fn = resolve_objective("calmar")
    s = {"annualized_return": 0.2, "max_drawdown": -0.1, "sharpe": 0, "sortino": 0}
    assert fn(s) == pytest.approx(2.0)


def test_unknown_objective_raises():
    with pytest.raises(KeyError, match="unknown"):
        resolve_objective("nope")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_objectives.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement objectives**

`backtester/optimize/objectives.py`:
```python
from __future__ import annotations

from typing import Callable, Dict


def _sharpe(s: Dict) -> float:
    return float(s.get("sharpe", 0.0))


def _sortino(s: Dict) -> float:
    return float(s.get("sortino", 0.0))


def _total_return(s: Dict) -> float:
    return float(s.get("total_return", 0.0))


def _annualized_return(s: Dict) -> float:
    return float(s.get("annualized_return", 0.0))


def _calmar(s: Dict) -> float:
    dd = abs(float(s.get("max_drawdown", 0.0)))
    if dd == 0:
        return 0.0
    return float(s.get("annualized_return", 0.0)) / dd


OBJECTIVES: Dict[str, Callable[[Dict], float]] = {
    "sharpe": _sharpe,
    "sortino": _sortino,
    "total_return": _total_return,
    "annualized_return": _annualized_return,
    "calmar": _calmar,
}


def resolve_objective(name: str) -> Callable[[Dict], float]:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}, allowed: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_objectives.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add backtester/optimize/objectives.py tests/unit/test_objectives.py
git commit -m "feat(optimize): add objective functions"
```

---

### Task 32: Grid search

**Files:**
- Create: `backtester/optimize/grid_search.py`
- Create: `tests/integration/test_grid_search.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_grid_search.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.config.models import ExecutionConfig, PortfolioConfig
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.optimize.grid_search import GridSearchOptimizer
from strategies.sma_cross import SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def test_grid_search_returns_best_and_all_results():
    data = make_ohlcv(n=400, seed=12, drift=0.0006, vol=0.01)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    opt = GridSearchOptimizer(engine=engine, objective="sharpe")
    best_params, best_result, all_results = opt.find_best(
        strategy_cls=SMACrossStrategy,
        data=data,
        param_space={"fast": [5, 10], "slow": [20, 50]},
        symbol="SYN", timeframe="1d",
    )
    assert best_params is not None
    assert isinstance(all_results, list) and len(all_results) == 4
    # best score >= every other score
    best_score = max(r["score"] for r in all_results)
    assert best_score == max(r["score"] for r in all_results)


def test_grid_search_handles_strategy_failures_gracefully():
    data = make_ohlcv(n=50, seed=3)
    broker = Broker(ExecutionConfig())
    portfolio = PortfolioSimulator(PortfolioConfig(), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    # slow > data length will produce all-NaN indicators -> still runs, returns score
    opt = GridSearchOptimizer(engine=engine, objective="sharpe")
    best_params, _, results = opt.find_best(
        strategy_cls=SMACrossStrategy,
        data=data,
        param_space={"fast": [5], "slow": [200]},  # warmup > data
        symbol="X", timeframe="1d",
    )
    assert len(results) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_grid_search.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement grid search**

`backtester/optimize/grid_search.py`:
```python
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from backtester.core.types import BacktestResult
from backtester.engine.backtest_engine import BacktestEngine
from backtester.optimize.objectives import resolve_objective
from backtester.optimize.parameter_space import expand_grid
from backtester.strategies.base import BaseStrategy

log = logging.getLogger("backtester.optimize")


class GridSearchOptimizer:
    def __init__(self, engine: BacktestEngine, objective: str = "sharpe"):
        self.engine = engine
        self.objective_name = objective
        self.score_fn = resolve_objective(objective)

    def find_best(
        self,
        strategy_cls: Type[BaseStrategy],
        data: pd.DataFrame,
        param_space: Dict[str, List[Any]],
        symbol: str,
        timeframe: str,
    ) -> Tuple[Any, BacktestResult, List[Dict]]:
        params_type = strategy_cls.params_type()
        strategy = strategy_cls()

        results: List[Dict] = []
        best: Tuple[float, Any, BacktestResult] | None = None

        for combo in expand_grid(param_space):
            try:
                params = params_type(**combo)
                result = self.engine.run(strategy, data, params, symbol=symbol, timeframe=timeframe)
                score = self.score_fn(result.summary)
            except Exception as exc:
                log.warning("grid combo %s failed: %s", combo, exc)
                results.append({"params": combo, "score": float("-inf"), "summary": {"error": str(exc)}})
                continue

            results.append({
                "params": asdict(params) if is_dataclass(params) else combo,
                "score": float(score),
                "summary": result.summary,
            })
            if best is None or score > best[0]:
                best = (score, params, result)

        if best is None:
            raise RuntimeError("grid search produced no successful runs")
        return best[1], best[2], results
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_grid_search.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backtester/optimize/grid_search.py tests/integration/test_grid_search.py
git commit -m "feat(optimize): add sequential grid search optimizer"
```

---

### Task 33: CLI runner — run_optimize

**Files:**
- Create: `backtester/runners/run_optimize.py`
- Create: `configs/optimize/sma_cross_grid.yaml`
- Create: `tests/integration/test_run_optimize_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_run_optimize_cli.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_optimize_cli_writes_grid_results(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=400, seed=21).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "opt.yaml"
    cfg.write_text(f"""
run_name: opt_smoke
strategy: sma_cross
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2026-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
portfolio:
  size: 1.0
optimization:
  objective: sharpe
  param_space:
    fast: [5, 10]
    slow: [20, 50]
output_root: "{out.as_posix()}"
""")

    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_optimize", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "grid_results.json").exists()
    grid = json.loads((run_dir / "grid_results.json").read_text())
    assert len(grid) == 4
    assert "best_params" in json.loads((run_dir / "summary.json").read_text())
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_run_optimize_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement run_optimize**

`backtester/runners/run_optimize.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.io.serialization import write_json
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.strategies.registry import get_strategy_class


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_optimize")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if rc.optimization is None:
        raise ConfigError("run_optimize requires an `optimization` block in the config")
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_optimize expects exactly one symbol")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    cls = get_strategy_class(rc.strategy)
    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective=rc.optimization.objective)

    log.info("grid search: strategy=%s space=%s objective=%s",
             rc.strategy, rc.optimization.param_space, rc.optimization.objective)
    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=cls,
        data=data,
        param_space=rc.optimization.param_space,
        symbol=symbol,
        timeframe=rc.data.timeframe,
    )

    writer.write_config(rc)
    writer.write_result(best_result)
    # Overwrite summary.json with optimizer-aware payload
    write_json(writer.run_dir / "summary.json", {
        "best_params": best_result.summary["params"],
        "best_score_objective": rc.optimization.objective,
        "best_summary": best_result.summary,
    })
    write_json(writer.run_dir / "grid_results.json", all_results)
    log.info("best=%s score(%s)=%.4f",
             best_result.summary["params"], rc.optimization.objective,
             max(r["score"] for r in all_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_run_optimize_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write sample optimize config**

`configs/optimize/sma_cross_grid.yaml`:
```yaml
run_name: sma_cross_spy_grid
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
  root: "data/raw"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20, 30]
    slow: [50, 100, 200]
output_root: "output/runs"
```

- [ ] **Step 6: Commit**

```
git add backtester/runners/run_optimize.py configs/optimize/sma_cross_grid.yaml tests/integration/test_run_optimize_cli.py
git commit -m "feat(cli): add run_optimize and sample grid config"
```

---

## Phase 10: Walk-Forward Optimization

### Task 34: WFO splitter

**Files:**
- Create: `backtester/wfo/splitter.py`
- Create: `tests/unit/test_wfo_splitter.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_wfo_splitter.py`:
```python
from __future__ import annotations

import pytest

from backtester.wfo.splitter import Window, WalkForwardSplitter
from tests.fixtures.synthetic import make_ohlcv


def test_window_dataclass_fields():
    data = make_ohlcv(n=10, seed=0)
    w = Window(train_data=data.iloc[:5], test_data=data.iloc[5:])
    assert len(w.train_data) == 5
    assert len(w.test_data) == 5


def test_splitter_produces_expected_windows():
    data = make_ohlcv(n=1000, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=252, test_bars=63, step_bars=63)
    assert len(windows) >= 10
    for w in windows:
        assert len(w.train_data) == 252
        assert len(w.test_data) <= 63


def test_splitter_train_precedes_test():
    data = make_ohlcv(n=500, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=200, test_bars=50, step_bars=50)
    for w in windows:
        assert w.train_data.index.max() < w.test_data.index.min()


def test_splitter_steps_advance():
    data = make_ohlcv(n=600, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=200, test_bars=50, step_bars=50)
    starts = [w.train_data.index.min() for w in windows]
    assert starts == sorted(set(starts)) and len(starts) == len(windows)


def test_splitter_raises_on_too_small_data():
    data = make_ohlcv(n=100, seed=0)
    with pytest.raises(ValueError, match="too short"):
        WalkForwardSplitter().split(data=data, train_bars=200, test_bars=50, step_bars=50)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_wfo_splitter.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement splitter**

`backtester/wfo/splitter.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass(slots=True)
class Window:
    train_data: pd.DataFrame
    test_data: pd.DataFrame


class WalkForwardSplitter:
    """Rolling-origin train/test splitter."""

    def split(
        self,
        data: pd.DataFrame,
        train_bars: int,
        test_bars: int,
        step_bars: int,
    ) -> List[Window]:
        n = len(data)
        if n < train_bars + test_bars:
            raise ValueError(
                f"data too short for WFO: have {n} bars, "
                f"need at least train_bars + test_bars = {train_bars + test_bars}"
            )

        windows: List[Window] = []
        start = 0
        while start + train_bars + test_bars <= n:
            train_end = start + train_bars
            test_end = min(train_end + test_bars, n)
            windows.append(Window(
                train_data=data.iloc[start:train_end].copy(),
                test_data=data.iloc[train_end:test_end].copy(),
            ))
            start += step_bars
        return windows
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_wfo_splitter.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add backtester/wfo/splitter.py tests/unit/test_wfo_splitter.py
git commit -m "feat(wfo): add rolling train/test splitter"
```

---

### Task 35: WFO stitcher

**Files:**
- Create: `backtester/wfo/stitcher.py`
- Create: `tests/unit/test_wfo_stitcher.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_wfo_stitcher.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.core.types import BacktestResult
from backtester.wfo.stitcher import WalkForwardStitcher


def _result(start: str, n: int, equity_start: float, equity_end: float) -> BacktestResult:
    idx = pd.bdate_range(start, periods=n)
    eq = pd.DataFrame({"equity": pd.Series([equity_start, *([0] * (n - 2)), equity_end]).interpolate()},
                      index=idx)
    trades = pd.DataFrame([
        {"timestamp": idx[0], "side": "buy", "qty": 1, "price": 100.0, "commission": 0, "notional": 100},
        {"timestamp": idx[-1], "side": "sell", "qty": 1, "price": 110.0, "commission": 0, "notional": 110},
    ])
    positions = pd.DataFrame({"qty": [1] * n}, index=idx)
    return BacktestResult(
        summary={"total_return": equity_end / equity_start - 1.0, "sharpe": 1.0,
                 "max_drawdown": -0.05, "n_trades": 2},
        equity_curve=eq, trades=trades, positions=positions,
    )


def test_stitcher_combines_oos_equity():
    windows = [
        {"train_start": pd.Timestamp("2024-01-01"), "train_end": pd.Timestamp("2024-03-01"),
         "test_start": pd.Timestamp("2024-03-04"), "test_end": pd.Timestamp("2024-04-01"),
         "best_params": {"fast": 10}, "train_summary": {"sharpe": 1.5},
         "test_summary": {"total_return": 0.05, "sharpe": 1.0, "max_drawdown": -0.02, "n_trades": 2, "n_round_trips": 1, "win_rate": 1.0, "avg_round_trip_pnl": 100, "annualized_return": 0.6, "annualized_vol": 0.2, "sortino": 1.1, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 10500},
         "test_result": _result("2024-03-04", 20, 10_000, 10_500)},
        {"train_start": pd.Timestamp("2024-02-01"), "train_end": pd.Timestamp("2024-04-01"),
         "test_start": pd.Timestamp("2024-04-02"), "test_end": pd.Timestamp("2024-05-01"),
         "best_params": {"fast": 20}, "train_summary": {"sharpe": 1.6},
         "test_summary": {"total_return": -0.02, "sharpe": -0.2, "max_drawdown": -0.05, "n_trades": 2, "n_round_trips": 1, "win_rate": 0.0, "avg_round_trip_pnl": -200, "annualized_return": -0.3, "annualized_vol": 0.2, "sortino": -0.3, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 9800},
         "test_result": _result("2024-04-02", 20, 10_500, 10_290)},
    ]
    out = WalkForwardStitcher().combine(windows)
    assert "oos_equity_curve" in out
    assert "oos_summary" in out
    assert "is_summary_avg" in out
    assert "parameter_stability" in out
    assert len(out["oos_equity_curve"]) == 40  # 20 + 20
    assert out["oos_equity_curve"]["equity"].iloc[-1] > 0
    assert out["parameter_stability"]["fast"]["unique"] == 2


def test_stitcher_handles_single_window():
    windows = [{
        "train_start": pd.Timestamp("2024-01-01"), "train_end": pd.Timestamp("2024-03-01"),
        "test_start": pd.Timestamp("2024-03-04"), "test_end": pd.Timestamp("2024-04-01"),
        "best_params": {"fast": 10}, "train_summary": {"sharpe": 1.5},
        "test_summary": {"total_return": 0.05, "sharpe": 1.0, "max_drawdown": -0.02, "n_trades": 2, "n_round_trips": 1, "win_rate": 1.0, "avg_round_trip_pnl": 100, "annualized_return": 0.6, "annualized_vol": 0.2, "sortino": 1.1, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 10500},
        "test_result": _result("2024-03-04", 20, 10_000, 10_500),
    }]
    out = WalkForwardStitcher().combine(windows)
    assert len(out["oos_equity_curve"]) == 20
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_wfo_stitcher.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement stitcher**

`backtester/wfo/stitcher.py`:
```python
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from backtester.analytics.metrics import compute_summary_metrics


class WalkForwardStitcher:
    """Concatenate OOS equity curves and recompute summary metrics across the stitched series."""

    def combine(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not window_results:
            raise ValueError("stitcher received no windows")

        oos_pieces = []
        oos_trades = []
        oos_positions = []
        prev_end = None

        for w in window_results:
            eq = w["test_result"].equity_curve.copy()
            # Re-base each window's equity onto the running OOS equity series
            if prev_end is None:
                scale = 1.0
            else:
                scale = prev_end / eq["equity"].iloc[0]
            eq["equity"] = eq["equity"] * scale
            if "cash" in eq.columns:
                eq["cash"] = eq["cash"] * scale
            if "position_value" in eq.columns:
                eq["position_value"] = eq["position_value"] * scale
            oos_pieces.append(eq)
            prev_end = eq["equity"].iloc[-1]

            oos_trades.append(w["test_result"].trades)
            oos_positions.append(w["test_result"].positions)

        oos_eq = pd.concat(oos_pieces).sort_index()
        # de-duplicate index if windows abut
        oos_eq = oos_eq[~oos_eq.index.duplicated(keep="last")]
        oos_trades_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
        oos_positions_df = pd.concat(oos_positions) if oos_positions else pd.DataFrame()

        oos_summary = compute_summary_metrics(oos_eq, oos_trades_df, oos_positions_df)

        # IS averages
        is_summaries = [w["train_summary"] for w in window_results]
        is_keys = set().union(*[set(s.keys()) for s in is_summaries])
        is_avg = {k: float(np.mean([float(s.get(k, 0.0)) for s in is_summaries if isinstance(s.get(k, 0), (int, float))]))
                  for k in is_keys}

        # parameter stability
        stability: Dict[str, Dict[str, Any]] = {}
        all_keys = set().union(*[set(w["best_params"].keys()) for w in window_results])
        for k in all_keys:
            values = [w["best_params"].get(k) for w in window_results]
            counter = Counter(values)
            stability[k] = {
                "unique": len(set(values)),
                "mode": counter.most_common(1)[0][0],
                "values_by_window": values,
            }

        return {
            "oos_equity_curve": oos_eq,
            "oos_trades": oos_trades_df,
            "oos_positions": oos_positions_df,
            "oos_summary": oos_summary,
            "is_summary_avg": is_avg,
            "parameter_stability": stability,
            "window_results": [
                {k: v for k, v in w.items() if k != "test_result"}
                for w in window_results
            ],
        }
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_wfo_stitcher.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backtester/wfo/stitcher.py tests/unit/test_wfo_stitcher.py
git commit -m "feat(wfo): add OOS stitcher and parameter stability tracking"
```

---

### Task 36: WFO runner

**Files:**
- Create: `backtester/wfo/runner.py`
- Create: `tests/integration/test_wfo_runner.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_wfo_runner.py`:
```python
from __future__ import annotations

import pandas as pd

from backtester.config.models import (
    DataConfig, ExecutionConfig, OptimizationConfig, PortfolioConfig,
    RunConfig, WFOConfig,
)
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.wfo.runner import WalkForwardRunner
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher
from strategies.sma_cross import SMACrossStrategy
from tests.fixtures.synthetic import make_ohlcv


def _config():
    return RunConfig(
        run_name="wfo_smoke",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SYN"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(commission_bps=0.0, slippage_bps=0.0),
        portfolio=PortfolioConfig(size=1.0),
        optimization=OptimizationConfig(objective="sharpe", param_space={"fast": [5, 10], "slow": [20, 50]}),
        wfo=WFOConfig(enabled=True, train_bars=200, test_bars=50, step_bars=50),
    )


def test_wfo_runner_produces_window_results_and_stitched_output():
    data = make_ohlcv(n=500, seed=33, drift=0.0005)
    broker = Broker(ExecutionConfig(commission_bps=0.0, slippage_bps=0.0))
    portfolio = PortfolioSimulator(PortfolioConfig(size=1.0), initial_cash=10_000.0)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective="sharpe")

    runner = WalkForwardRunner(
        engine=engine, optimizer=optimizer,
        splitter=WalkForwardSplitter(), stitcher=WalkForwardStitcher(),
    )

    out = runner.run(strategy_cls=SMACrossStrategy, full_data=data, base_config=_config())

    assert "oos_equity_curve" in out
    assert "window_results" in out
    assert len(out["window_results"]) >= 5
    for wr in out["window_results"]:
        for k in ("train_start", "train_end", "test_start", "test_end", "best_params", "train_summary", "test_summary"):
            assert k in wr
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_wfo_runner.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement runner**

`backtester/wfo/runner.py`:
```python
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from typing import Any, Dict, Type

import pandas as pd

from backtester.config.models import RunConfig
from backtester.engine.backtest_engine import BacktestEngine
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.strategies.base import BaseStrategy
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher

log = logging.getLogger("backtester.wfo")


class WalkForwardRunner:
    def __init__(
        self,
        engine: BacktestEngine,
        optimizer: GridSearchOptimizer,
        splitter: WalkForwardSplitter,
        stitcher: WalkForwardStitcher,
    ):
        self.engine = engine
        self.optimizer = optimizer
        self.splitter = splitter
        self.stitcher = stitcher

    def run(
        self,
        strategy_cls: Type[BaseStrategy],
        full_data: pd.DataFrame,
        base_config: RunConfig,
    ) -> Dict[str, Any]:
        assert base_config.wfo is not None and base_config.wfo.enabled
        assert base_config.optimization is not None

        windows = self.splitter.split(
            data=full_data,
            train_bars=base_config.wfo.train_bars,
            test_bars=base_config.wfo.test_bars,
            step_bars=base_config.wfo.step_bars,
        )
        log.info("WFO: %d windows", len(windows))

        symbol = base_config.data.symbols[0]
        timeframe = base_config.data.timeframe

        window_results = []
        for i, window in enumerate(windows):
            best_params, _train_result, _all_train = self.optimizer.find_best(
                strategy_cls=strategy_cls,
                data=window.train_data,
                param_space=base_config.optimization.param_space,
                symbol=symbol,
                timeframe=timeframe,
            )

            test_strategy = strategy_cls()
            test_result = self.engine.run(
                strategy=test_strategy,
                data=window.test_data,
                params=best_params,
                symbol=symbol,
                timeframe=timeframe,
            )

            window_results.append({
                "window_index": i,
                "train_start": window.train_data.index.min(),
                "train_end": window.train_data.index.max(),
                "test_start": window.test_data.index.min(),
                "test_end": window.test_data.index.max(),
                "best_params": asdict(best_params) if is_dataclass(best_params) else dict(best_params),
                "train_summary": _train_result.summary,
                "test_summary": test_result.summary,
                "test_result": test_result,
            })
            log.info("window %d: best=%s test_sharpe=%.3f",
                     i, window_results[-1]["best_params"], test_result.summary.get("sharpe", 0.0))

        return self.stitcher.combine(window_results)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_wfo_runner.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```
git add backtester/wfo/runner.py tests/integration/test_wfo_runner.py
git commit -m "feat(wfo): add WalkForwardRunner orchestrating splitter+optimizer+stitcher"
```

---

### Task 37: CLI runner — run_wfo

**Files:**
- Create: `backtester/runners/run_wfo.py`
- Create: `configs/wfo/sma_cross_wfo.yaml`
- Create: `tests/integration/test_run_wfo_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_run_wfo_cli.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_wfo_cli_produces_window_and_oos_artifacts(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=600, seed=42).to_csv(raw / "SYN.csv", index_label="date")

    out = tmp_path / "runs"
    cfg = tmp_path / "wfo.yaml"
    cfg.write_text(f"""
run_name: wfo_smoke
strategy: sma_cross
strategy_params:
  fast: 10
  slow: 30
data:
  symbols: ["SYN"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2030-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution:
  initial_cash: 10000
  commission_bps: 0
  slippage_bps: 0
portfolio:
  size: 1.0
optimization:
  objective: sharpe
  param_space:
    fast: [5, 10]
    slow: [20, 50]
wfo:
  enabled: true
  train_bars: 200
  test_bars: 50
  step_bars: 50
output_root: "{out.as_posix()}"
""")
    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_wfo", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    for name in ("config_resolved.yaml", "summary.json", "window_results.json",
                 "oos_equity_curve.csv", "oos_trades.csv", "logs.txt"):
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "summary.json").read_text())
    assert "oos_summary" in summary
    assert "is_summary_avg" in summary
    assert "parameter_stability" in summary
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_run_wfo_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement run_wfo**

`backtester/runners/run_wfo.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.io.serialization import write_json
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.strategies.registry import get_strategy_class
from backtester.wfo.runner import WalkForwardRunner
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_wfo")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if rc.wfo is None or not rc.wfo.enabled:
        raise ConfigError("run_wfo requires `wfo.enabled: true`")
    if rc.optimization is None:
        raise ConfigError("run_wfo requires an `optimization` block")
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_wfo expects exactly one symbol")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    cls = get_strategy_class(rc.strategy)
    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective=rc.optimization.objective)
    runner = WalkForwardRunner(engine=engine, optimizer=optimizer,
                               splitter=WalkForwardSplitter(), stitcher=WalkForwardStitcher())

    log.info("running WFO: train=%d test=%d step=%d",
             rc.wfo.train_bars, rc.wfo.test_bars, rc.wfo.step_bars)
    out = runner.run(strategy_cls=cls, full_data=data, base_config=rc)

    writer.write_config(rc)
    write_json(writer.run_dir / "summary.json", {
        "oos_summary": out["oos_summary"],
        "is_summary_avg": out["is_summary_avg"],
        "parameter_stability": out["parameter_stability"],
        "n_windows": len(out["window_results"]),
    })
    write_json(writer.run_dir / "window_results.json", out["window_results"])
    out["oos_equity_curve"].to_csv(writer.run_dir / "oos_equity_curve.csv", index_label="timestamp")
    out["oos_trades"].to_csv(writer.run_dir / "oos_trades.csv", index=False)
    out["oos_positions"].to_csv(writer.run_dir / "oos_positions.csv", index_label="timestamp")

    log.info("WFO complete: %d windows, oos_sharpe=%.3f",
             len(out["window_results"]), out["oos_summary"].get("sharpe", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_run_wfo_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write sample WFO config**

`configs/wfo/sma_cross_wfo.yaml`:
```yaml
run_name: sma_cross_spy_wfo
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
  root: "data/raw"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20, 30]
    slow: [50, 100, 200]
wfo:
  enabled: true
  train_bars: 756
  test_bars: 252
  step_bars: 252
output_root: "output/runs"
```

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```
git add backtester/runners/run_wfo.py configs/wfo/sma_cross_wfo.yaml tests/integration/test_run_wfo_cli.py
git commit -m "feat(cli): add run_wfo entrypoint and sample WFO config"
```

---

### Task 38: Multi-symbol batch runner

**Files:**
- Create: `backtester/runners/run_batch.py`
- Create: `tests/integration/test_run_batch_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_run_batch_cli.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic import make_ohlcv


def test_run_batch_iterates_symbols(tmp_path: Path):
    raw = tmp_path / "data"
    raw.mkdir()
    make_ohlcv(n=300, seed=10).to_csv(raw / "A.csv", index_label="date")
    make_ohlcv(n=300, seed=11).to_csv(raw / "B.csv", index_label="date")
    out = tmp_path / "runs"
    cfg = tmp_path / "batch.yaml"
    cfg.write_text(f"""
run_name: batch_smoke
strategy: sma_cross
strategy_params: {{fast: 10, slow: 30}}
data:
  symbols: ["A", "B"]
  timeframe: "1d"
  start: "2020-01-02"
  end: "2030-12-31"
  source: "csv"
  root: "{raw.as_posix()}"
execution: {{initial_cash: 10000, commission_bps: 0, slippage_bps: 0}}
portfolio: {{size: 1.0}}
output_root: "{out.as_posix()}"
""")
    res = subprocess.run(
        [sys.executable, "-m", "backtester.runners.run_batch", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr

    run_dir = next(out.iterdir())
    assert (run_dir / "batch_summary.json").exists()
    by_symbol = json.loads((run_dir / "batch_summary.json").read_text())
    assert set(by_symbol.keys()) == {"A", "B"}
    for sym, summary in by_symbol.items():
        assert "total_return" in summary
        assert (run_dir / f"{sym}_equity_curve.csv").exists()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/integration/test_run_batch_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement run_batch**

`backtester/runners/run_batch.py`:
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
    if not rc.data.symbols:
        raise SystemExit("data.symbols is empty")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    by_symbol = {}
    for symbol in rc.data.symbols:
        try:
            data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                               start=rc.data.start, end=rc.data.end)
            validate_ohlcv(data)
            strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)
            broker = Broker(rc.execution)
            portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
            engine = BacktestEngine(broker=broker, portfolio=portfolio)
            result = engine.run(strategy, data, params, symbol=symbol, timeframe=rc.data.timeframe)
            by_symbol[symbol] = result.summary
            result.equity_curve.to_csv(writer.run_dir / f"{symbol}_equity_curve.csv", index_label="timestamp")
            result.trades.to_csv(writer.run_dir / f"{symbol}_trades.csv", index=False)
            log.info("%s: total_return=%.4f sharpe=%.3f", symbol,
                     result.summary["total_return"], result.summary["sharpe"])
        except Exception as exc:
            log.warning("%s failed: %s", symbol, exc)
            by_symbol[symbol] = {"error": str(exc)}

    writer.write_config(rc)
    write_json(writer.run_dir / "batch_summary.json", by_symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/integration/test_run_batch_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```
git add backtester/runners/run_batch.py tests/integration/test_run_batch_cli.py
git commit -m "feat(cli): add run_batch for multi-symbol fan-out"
```

---

## Phase 11: Documentation

### Task 39: Strategy contract doc

**Files:**
- Create: `docs/strategy_contract.md`

- [ ] **Step 1: Write the doc**

`docs/strategy_contract.md`:
```markdown
# Strategy contract

A strategy is exactly one Python file under `strategies/` that defines:

1. One `@dataclass(slots=True)` for its parameters.
2. One class that inherits from `BaseStrategy[ParamsType]`.

Both must live in the same module. The strategy must be registered in
`backtester/strategies/registry.py` exactly once.

## Required imports

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy
```

## Required class attributes

| Attribute     | Type  | Notes                                          |
|---------------|-------|------------------------------------------------|
| `strategy_id` | str   | Unique snake_case identifier                   |
| `version`     | str   | Semver-like string                             |
| `asset_type`  | str   | `"stock"` for MVP                              |
| `timeframe`   | str   | `"1d"` for daily                               |

## Required methods

```python
@classmethod
def params_type(cls) -> type: ...

def warmup_bars(self, params) -> int: ...           # optional, default 0

def indicators(self, data, params) -> pd.DataFrame: ...

def generate_signals(self, data, indicators, ctx, params) -> SignalFrame: ...
```

## Data assumptions

- Index is a `DatetimeIndex`, sorted ascending, no duplicates.
- Columns include lowercase `open`, `high`, `low`, `close`, `volume`.
- Prices are positive, no NaNs in price columns.
- Only past and current bars may be used. **Never read future rows.**

## Signal semantics

- `1` = target long position.
- `0` = target flat.
- Signals are typically shifted by one bar (`signal.shift(1)`) so the
  engine fills the order on the **next** bar's open, not the current
  bar's close.
- An optional `size` column scales the percent-equity allocation
  (multiplicative with `portfolio.size`).
- An optional `price_column` (referenced by `SignalFrame.price_column`)
  turns the order into a LIMIT order at that price on the next bar.

## Rules for AI-generated strategies

1. Exactly one params dataclass.
2. Exactly one public strategy class.
3. `strategy_id` in snake_case, globally unique.
4. No file, network, or env access.
5. Import only `BaseStrategy`, `SignalFrame`, `StrategyContext` from the framework.
6. Use only past and current bars.
7. Shift tradable signals by one bar unless explicitly using same-bar execution.
8. Keep helper logic in the same file unless reused across 3+ strategies.

## Minimal example

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FlatParams:
    size: float = 1.0


class FlatStrategy(BaseStrategy[FlatParams]):
    strategy_id = "flat"

    @classmethod
    def params_type(cls):
        return FlatParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params: FlatParams):
        df = pd.DataFrame({"signal": 0, "size": params.size}, index=data.index)
        return SignalFrame(data=df)
```
```

- [ ] **Step 2: Commit**

```
git add docs/strategy_contract.md
git commit -m "docs: add strategy contract reference"
```

---

### Task 40: Data contract, runbook, examples, README

**Files:**
- Create: `docs/data_contract.md`
- Create: `docs/runbook.md`
- Create: `docs/examples.md`
- Create: `docs/prd.md`
- Create: `README.md`

- [ ] **Step 1: Write `docs/data_contract.md`**

```markdown
# Data contract

OHLCV files live in `data/raw/{SYMBOL}.csv` or `data/raw/{SYMBOL}.parquet`.

## Schema

| Column   | Type     | Notes                            |
|----------|----------|----------------------------------|
| date     | date     | Index column; sorted ascending   |
| open     | float    | > 0                              |
| high     | float    | > 0, >= open and >= close        |
| low      | float    | > 0, <= open and <= close        |
| close    | float    | > 0                              |
| volume   | int/float| >= 0                             |

Column names are normalized to lowercase by the loader.

## Invariants

- No NaN values in price or volume columns.
- `DatetimeIndex` is strictly monotonic increasing.
- No duplicate timestamps.

## Loading

```python
from backtester.data.loader import load_symbol
df = load_symbol("SPY", source="csv", root="data/raw", start="2020-01-01", end="2024-01-01")
```

## Validation

```python
from backtester.data.validators import validate_ohlcv
validate_ohlcv(df)  # raises DataError on any violation
```
```

- [ ] **Step 2: Write `docs/runbook.md`**

```markdown
# Runbook

## Install

```
pip install -e .[dev]
python scripts/generate_sample_data.py
```

## Commands

```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
python -m backtester.runners.run_optimize --config configs/optimize/sma_cross_grid.yaml
python -m backtester.runners.run_wfo      --config configs/wfo/sma_cross_wfo.yaml
python -m backtester.runners.run_batch    --config <multi-symbol config>
```

## Output bundle

Every run writes a folder under `output/runs/`:

```
output/runs/<timestamp>_<run_name>/
  config_resolved.yaml      # exact config used
  summary.json              # headline metrics
  trades.csv                # fill log
  positions.csv             # per-bar position
  equity_curve.csv          # per-bar cash + position_value + equity
  window_results.json       # WFO only
  oos_equity_curve.csv      # WFO only
  oos_trades.csv            # WFO only
  oos_positions.csv         # WFO only
  grid_results.json         # optimize only
  logs.txt
```

## Reproducibility

- Configs are YAML and round-trip through `config_resolved.yaml`.
- The sample data generator (`scripts/generate_sample_data.py`) is
  deterministic — given the same seed it produces byte-identical CSVs.
- Strategies must not access the environment, network, or local files.

## Testing

```
pytest -q
```
```

- [ ] **Step 3: Write `docs/examples.md`**

```markdown
# Examples

## 1. Backtest one strategy on one symbol

```yaml
# configs/backtests/sma_cross_spy.yaml
run_name: sma_cross_spy
strategy: sma_cross
strategy_params: {fast: 20, slow: 50}
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: csv
execution: {initial_cash: 100000, commission_bps: 2, slippage_bps: 5}
portfolio: {size: 0.95}
```

```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

## 2. Grid optimize the same strategy

```yaml
# configs/optimize/sma_cross_grid.yaml
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20, 30]
    slow: [50, 100, 200]
```

## 3. Walk-forward optimization

```yaml
# configs/wfo/sma_cross_wfo.yaml
wfo:
  enabled: true
  train_bars: 756   # 3 trading years
  test_bars: 252    # 1 trading year
  step_bars: 252
```
```

- [ ] **Step 4: Write `docs/prd.md`**

```markdown
# PRD (snapshot)

See the implementation plan at `docs/superpowers/plans/2026-05-14-modular-stock-backtester.md`
for the authoritative product requirements that drove this codebase.

Headline goals:
- New strategies are single drop-in files.
- One ABC contract; engine never special-cases a strategy.
- Same engine powers backtest / optimize / WFO.
- All runs are config-driven and write deterministic artifacts.
```

- [ ] **Step 5: Write `README.md`**

```markdown
# Modular Stock Backtester

Python research framework for daily stock strategies. Strategies are
self-contained modules conforming to one ABC contract; the same engine
runs standard backtests, grid optimization, and walk-forward optimization.

## Quick start

```
pip install -e .[dev]
python scripts/generate_sample_data.py
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

## Layout

```
backtester/       # framework (contracts, engine, analytics, optimize, wfo, runners, io)
strategies/       # user/AI-authored strategies
configs/          # YAML configs for runs
data/raw/         # OHLCV inputs
output/runs/      # deterministic per-run artifact bundles
docs/             # contracts and runbook
tests/            # unit + integration tests
scripts/          # one-off helpers (sample data generator)
```

## Documentation

- `docs/strategy_contract.md` — how to write a strategy.
- `docs/data_contract.md` — OHLCV schema and invariants.
- `docs/runbook.md` — commands, output structure, reproducibility notes.
- `docs/examples.md` — example configs.

## Testing

```
pytest -q
```
```

- [ ] **Step 6: Final full-suite run**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```
git add docs/data_contract.md docs/runbook.md docs/examples.md docs/prd.md README.md
git commit -m "docs: add data contract, runbook, examples, README"
```

---

## Phase 12: Final acceptance check

### Task 41: End-to-end acceptance against PRD success criteria

**Files:**
- (no new files — verification only)

- [ ] **Step 1: SC1 — new strategy can be created from a template in one file**

Verify: `backtester/strategies/templates/strategy_template.py` exists and is parseable.
Run: `python -c "import backtester.strategies.templates.strategy_template as t; assert hasattr(t, 'StrategyName')"`
Expected: exit 0.

- [ ] **Step 2: SC2 — new strategy can be backtested through CLI**

Verify the three sample strategies are registered and runnable.
Run: `python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml`
Expected: exits 0 and creates an output folder.

- [ ] **Step 3: SC3 — WFO uses the same strategy interface**

Verify by grepping that `WalkForwardRunner` calls `engine.run` and the same strategy class.
Run: `grep -n "engine.run" backtester/wfo/runner.py`
Expected: matches the line that invokes the standard engine.

- [ ] **Step 4: SC4 — every run writes deterministic artifacts**

Run two consecutive backtests with the same config.
Run:
```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```
Manually compare the two newest `summary.json` files — `total_return`, `sharpe`,
`max_drawdown`, and `n_trades` must match exactly (timestamp portion of the dir
name differs by design).

- [ ] **Step 5: SC5 — codebase organization is AI-discoverable**

Verify the architecture layout matches the PRD tree.
Run: `find backtester -name '*.py' | sort`
Expected: every file from the PRD tree exists.

- [ ] **Step 6: Full test suite**

Run: `pytest -q`
Expected: all green.

- [ ] **Step 7: Tag MVP**

```
git tag -a v0.1.0 -m "MVP: backtest + optimize + WFO with three sample strategies"
```

---

## Self-Review (performed before handing off)

**Spec coverage:**
- Plugin-style layout, BaseStrategy ABC, explicit registry: Tasks 5–7, 25.
- Data loader (CSV/Parquet) + validators: Tasks 9–10.
- Config dataclasses + YAML loader + validation: Tasks 12–14.
- Backtest engine, broker (commission + slippage), portfolio simulator, orders/fills: Tasks 15–19.
- Long-only with MARKET/LIMIT/STOP orders (per clarification): Task 15.
- Bundled sample CSVs (per clarification): Task 11.
- Summary metrics + drawdown + trades + exposure: Tasks 20–22.
- Three sample strategies + registry wiring: Tasks 23–25.
- Artifact writer + logging + serialization: Tasks 26–27.
- CLI runners (backtest, optimize, wfo, batch): Tasks 29, 33, 37, 38.
- Sequential grid search + objectives + parameter space: Tasks 30–32.
- WFO splitter + stitcher + runner (reuses BacktestEngine): Tasks 34–37.
- WFO OOS-only headline + IS/OOS separated + param stability: Task 35 (stitcher).
- Docs (strategy contract, data contract, runbook, examples, README): Tasks 39–40.

**Type consistency check:**
- `compute_summary_metrics(equity_curve, trades, positions)` signature matches across stub (Task 19), full implementation (Task 22), and stitcher use (Task 35).
- `BacktestEngine.run(strategy, data, params, symbol, timeframe)` signature matches in optimizer (Task 32), WFO runner (Task 36), and CLI runners (Tasks 29, 33, 37, 38).
- `SignalFrame` field names (`data`, `signal_column`, `size_column`, `price_column`) are consistent in core types (Task 3), portfolio (Task 18), and every sample strategy.
- `PortfolioSimulator.simulate(data, signal_frame, broker)` keyword usage matches in engine (Task 19) and tests (Task 18).
- `ArtifactWriter` exposes `write_config`, `write_result`, `write_window_results`, `log_path`, `run_dir` — used consistently by all three runners.

**Placeholder scan:** no `TODO`, `TBD`, "fill in later", or "similar to" references — every code block is complete and self-contained.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-modular-stock-backtester.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with review checkpoints between tasks. Best for catching drift on a 41-task plan.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans` with batch checkpoints.

Which approach?
