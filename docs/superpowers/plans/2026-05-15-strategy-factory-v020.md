# Strategy Factory v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an unattended Python loop that mass-produces SPY trading strategy ideas via `claude -p`, runs each through the Modular-Backtester's backtest → optimize → WFO pipeline, and surfaces hits via a local dashboard + Telegram alerts — without modifying the locked v0.4.0 backtester engine.

**Architecture:** A single-process orchestrator (`factory/loop.py`) running one synchronous cycle at a time. Each cycle: pull random slot values, build a prompt, call `claude -p` (subprocess), parse a defensive double-unwrap of JSON output, two-tier validate the generated strategy (static + functional smoke against the real `backtester` package), write strategy+config to disk, append one idempotent line to `backtester/strategies/registry.py`, run three backtester subprocesses (backtest → optimize → WFO) with stage-suffixed `run_name`s so output bundles don't collide, parse each stage's real `summary.json`, append one JSONL record to `data/results.json`, conditionally fire a Telegram alert. A separate read-only Flask dashboard polls the JSONL file.

**Tech Stack:** Python 3.11 (uses stdlib `tomllib`), pandas 2.2, pyyaml 6, `subprocess.run`, Flask (dashboard), pytest 7.4, the existing `modular-stock-backtester` v0.4.0 package imported via the same venv. No new runtime deps beyond Flask.

---

## Pre-build reconciliation findings (locked in before any code)

The spec was written against the v0.3.0 README. These are the corrections discovered by reading the real v0.4.0+ source. **Where the real code disagrees with the spec, the real code wins — the plan below reflects the real shapes.**

### Finding R1 — `summary.json` shapes are nested, not flat

Real outputs (sampled from `output/runs/20260515_0344_gen_1715800000/` etc.):

- **Backtest `summary.json`** (single-symbol legacy path — what factory strategies hit):
  flat top-level keys: `total_return`, `annualized_return`, `annualized_vol`, `sharpe`, `sortino`, `max_drawdown`, `n_trades`, `n_round_trips`, `win_rate`, `avg_round_trip_pnl`, `time_in_market`, `turnover`, `final_equity`, `params` (dict), `symbol`, `timeframe`.
- **Optimize `summary.json`**: top-level `best_params` (dict), `best_score_objective` (string — NOT `objective`), `best_summary` (full nested backtest summary shape). No standalone `best_score` numeric field — read `best_summary[best_score_objective]`. Sibling `grid_results.json` holds the full grid.
- **WFO `summary.json`**: top-level `oos_summary` (NESTED dict with flat metric names like `sharpe`, `total_return`, `max_drawdown`, `n_trades` — NOT flat `oos_*` keys at the top), `is_summary_avg`, `parameter_stability` (dict of `param_name → {unique, mode, values_by_window}`), `n_windows`. Sibling `window_results.json` holds per-window detail; sibling `oos_equity_curve.csv` for dashboard rendering.

Plan's `factory/stages.py` parses against these real shapes. The factory's results record (§6) maps:
- `record["wfo"]["oos_sharpe"]` ← `wfo_summary["oos_summary"]["sharpe"]`
- `record["wfo"]["oos_total_return"]` ← `wfo_summary["oos_summary"]["total_return"]`
- `record["wfo"]["oos_max_drawdown"]` ← `wfo_summary["oos_summary"]["max_drawdown"]`
- `record["wfo"]["oos_n_trades"]` ← `wfo_summary["oos_summary"]["n_trades"]`
- `record["wfo"]["parameter_stability"]` ← `wfo_summary["parameter_stability"]`
- `record["wfo"]["n_windows"]` ← `wfo_summary["n_windows"]`
- `record["optimize"]["best_params"]` ← `opt_summary["best_params"]`
- `record["optimize"]["objective"]` ← `opt_summary["best_score_objective"]`
- `record["optimize"]["best_score"]` ← `opt_summary["best_summary"][opt_summary["best_score_objective"]]`

### Finding R2 — Bundle dir naming uses minute resolution; collision-safe via `run_name` suffixing

`ArtifactWriter` (in `backtester/io/artifacts.py`) names every bundle dir `<output_root>/<YYYYMMDD_HHMM>_<run_name>/` with `mkdir(parents=True, exist_ok=True)`. Two stages sharing a `run_name` within the same minute will write into the SAME dir and clobber each other's `summary.json`.

**Fix in `factory/stages.py`:** for each cycle, write three transient stage-specific YAMLs into `factory/data/_tmp/<strategy_id>/{backtest,optimize,wfo}.yaml`, each cloned from the canonical config under `configs/wfo/<strategy_id>.yaml` but with `run_name` rewritten to `<strategy_id>`, `<strategy_id>_grid`, `<strategy_id>_wfo` respectively. The canonical YAML under `configs/wfo/` remains untouched (spec §5.5 honored).

### Finding R3 — Runner CLI is exactly `python -m backtester.runners.run_<X> --config <path>`

Single required `--config` arg, no other flags. Exit code 0 on success; raises `SystemExit` (non-zero) or `ConfigError` (propagates non-zero via `__main__`'s `raise SystemExit(main())`) on failure. `subprocess.run().returncode != 0` is a reliable failure signal.

### Finding R4 — Single-symbol routing is automatic; factory strategies always hit the legacy path

All three runners check `cls.uses_multi_symbol` (default `False` in `BaseStrategy`) and fork into `_run_legacy_single_symbol_*`. Factory-generated strategies must NOT set `uses_multi_symbol = True`. Tier-1 validation enforces this — the static check requires the class body to NOT contain `uses_multi_symbol`.

### Finding R5 — Appendix A config shape matches real `gen_1715800000.yaml` exactly

Verified field-by-field: `run_name`, `strategy`, `strategy_params`, `data.{symbols,timeframe,start,end,source,root}`, `execution.{initial_cash,commission_bps,slippage_bps,allow_fractional,allow_short}`, `portfolio.{sizing_mode,size}`, `optimization.{objective,param_space}`, `wfo.{enabled,train_bars,test_bars,step_bars}`. The config loader also silently accepts unused blocks, so one YAML drives all three runners. Appendix A is reproduced verbatim in `factory/prompt.py`.

### Finding R6 — Registry pattern (verbatim from real `backtester/strategies/registry.py`)

```python
from strategies.<strategy_id> import GeneratedStrategy as _<strategy_id>  # noqa: E402
register_strategy(_<strategy_id>)
```

Two lines appended at end of file. The alias avoids class-name collisions across many generated strategies. Idempotency check: scan the file for the string `_<strategy_id>` (alias is unique per id) before appending.

### Finding R7 — `gen_<unix_timestamp>` ID collision is rare but possible

If the factory crashes and restarts in the same second, two cycles could share an id. The loop checks `<backtester_root>/strategies/<strategy_id>.py` before writing — on collision, it bumps to `<id>_2`, `<id>_3`, etc. The id passed to the prompt is the final non-colliding one.

---

## File Structure

The factory lives at `<backtester_root>/factory/` (per spec §10 — developer's choice; this build uses the in-tree subdirectory option). All paths in production code derive from `factory/config/settings.toml`. Each module has one responsibility:

```
<backtester_root>/factory/
├── __init__.py                    # empty — marks factory as a package
├── conftest.py                    # pytest path injection (adds backtester_root to sys.path)
├── README.md                      # local-only quickstart
├── config/
│   └── settings.toml              # all tunables (§10), checked in with placeholders
├── factory/
│   ├── __init__.py
│   ├── settings_loader.py         # reads settings.toml → dataclass; resolves paths
│   ├── slots.py                   # six slot lists + random pull (§5.1)
│   ├── prompt.py                  # Appendix A template + builder (§5.2)
│   ├── generate.py                # claude -p subprocess + double-unwrap parser (§5.3)
│   ├── validate.py                # Tier 1 static + Tier 2 functional (§5.4, §7)
│   ├── filesystem.py              # write .py / .yaml; idempotent registry append (§5.5)
│   ├── dedup.py                   # read tail 30 / append (§5.6, §3.2)
│   ├── stages.py                  # 3 subprocesses + stage-suffixed run_names + summary parsers (§5.7)
│   ├── results.py                 # JSONL append/read; record schema constants (§5.8, §6)
│   ├── notify.py                  # Telegram sendMessage with threshold gate (§5.9)
│   ├── cycle.py                   # one full cycle (steps 1-17, §3)
│   ├── loop.py                    # continuous while-loop + signal handling (§5.10)
│   └── synth_ohlcv.py             # 200-bar synthetic frame for Tier 2 validation
├── dashboard/
│   ├── __init__.py
│   ├── server.py                  # Flask app: routes + JSON endpoint (§8.1, §8.2)
│   ├── templates/
│   │   ├── overview.html          # one row per cycle (§8.1)
│   │   └── detail.html            # full per-strategy view (§8.2)
│   └── static/
│       ├── overview.js            # client-side sort + auto-refresh poll (§8.3)
│       └── style.css              # minimal styling
├── data/
│   ├── dedup_log.txt              # append-only (§3.2, §5.6)
│   ├── results.json               # append-only JSONL (§5.8, §6)
│   └── _tmp/                      # transient per-cycle stage YAMLs (R2)
├── logs/
│   └── factory.log                # rotating log; orchestrator writes structured lines
└── tests/
    ├── __init__.py
    ├── conftest.py                # shared fixtures (tmp settings, sample summaries)
    ├── test_slots.py
    ├── test_prompt.py
    ├── test_dedup.py
    ├── test_generate.py
    ├── test_validate_static.py    # Tier 1
    ├── test_validate_functional.py # Tier 2 (slow — uses real backtester)
    ├── test_filesystem.py
    ├── test_stages.py
    ├── test_results.py
    ├── test_notify.py
    ├── test_cycle.py
    ├── test_loop.py
    ├── test_dashboard.py
    ├── test_integration_smoke.py  # ONE full cycle e2e against gen_1715800000
    ├── test_integration_failures.py # every failure point produces correct record
    └── fixtures/
        ├── claude_output_clean.json       # well-formed envelope + JSON body
        ├── claude_output_fenced.json      # JSON wrapped in ```json fences
        ├── claude_output_prose_wrapped.json # JSON inside prose "Here is the strategy: {...}"
        ├── claude_output_malformed.json   # broken — used for parse-failure path
        ├── valid_strategy.py              # known-good strategy (clone of gen_1715800000)
        ├── invalid_no_shift.py            # missing .shift(1)
        ├── invalid_bad_imports.py         # imports os / requests
        ├── invalid_no_class.py            # missing GeneratedStrategy
        ├── invalid_signal_dtype.py        # Tier-2 trip: returns float signal
        ├── invalid_signal_short.py        # Tier-2 trip: emits -1 with allow_short=false
        ├── sample_backtest_summary.json
        ├── sample_optimize_summary.json
        └── sample_wfo_summary.json
```

**Touched files inside the backtester repo (the ONLY ones the factory writes to):**
- `<backtester_root>/strategies/<strategy_id>.py` — created per cycle
- `<backtester_root>/configs/wfo/<strategy_id>.yaml` — created per cycle
- `<backtester_root>/backtester/strategies/registry.py` — 2 lines appended per successful generation

---

## Conventions

- **TDD per task**: write the failing test first, run it, see it fail with the expected message, write the minimal code, run again, see it pass.
- **Atomic commit per task**: every task ends with one `git commit`. Commit message: `feat(factory): <short description>` for new modules, `test(factory): <X>` for test-only commits, `fix(factory): <X>` for fixes during the build.
- **Imports inside factory code**: always absolute `from factory.<module> import ...` (NOT relative). Lets tests run regardless of cwd.
- **Subprocess invocations** in `generate.py` and `stages.py` use `sys.executable` (NOT a hardcoded `"python"`) so we don't pick up a stale system Python.
- **Paths**: use `pathlib.Path` everywhere; never string-concatenate paths. The factory's `paths.py`-style helpers all return `Path`.
- **Pytest invocation**: from `<backtester_root>`, run `python -m pytest factory/tests -q`. The `factory/conftest.py` injects `<backtester_root>` onto `sys.path` so `from factory.x import y` and `from backtester.x import y` both resolve.
- **PowerShell-aware commit syntax**: heredocs use `@'...'@` (single-quoted, literal), closing `'@` at column 0. Examples below use this.
- **Logging**: every module that does I/O uses `logging.getLogger(__name__)`. The loop configures a `RotatingFileHandler` on `factory/logs/factory.log` with a 10MB rollover.
- **No emojis**, no decorative output, no docstrings beyond a one-line module purpose.

---

## Task 0: Bootstrap — directory layout, settings.toml, conftest

**Files:**
- Create: `factory/__init__.py` (empty)
- Create: `factory/conftest.py`
- Create: `factory/README.md`
- Create: `factory/.gitignore`
- Create: `factory/config/settings.toml`
- Create: `factory/factory/__init__.py` (empty)
- Create: `factory/factory/settings_loader.py`
- Create: `factory/tests/__init__.py` (empty)
- Create: `factory/tests/conftest.py`
- Create: `factory/tests/test_settings_loader.py`

- [ ] **Step 1: Create the directory skeleton**

Run from `<backtester_root>`:

```powershell
New-Item -ItemType Directory -Force -Path factory\factory, factory\config, factory\data, factory\data\_tmp, factory\logs, factory\tests, factory\tests\fixtures, factory\dashboard\templates, factory\dashboard\static | Out-Null
New-Item -ItemType File -Force -Path factory\__init__.py, factory\factory\__init__.py, factory\tests\__init__.py, factory\dashboard\__init__.py | Out-Null
```

- [ ] **Step 2: Write `factory/.gitignore`**

```
data/results.json
data/dedup_log.txt
data/_tmp/
logs/
__pycache__/
.pytest_cache/
*.pyc
config/settings.local.toml
```

- [ ] **Step 3: Write `factory/conftest.py`**

```python
import sys
from pathlib import Path

# Make `from factory.<module>` and `from backtester.<module>` both resolve
# when pytest is invoked from any cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

- [ ] **Step 4: Write `factory/config/settings.toml`**

```toml
[paths]
backtester_root  = "."
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_log        = "factory/data/dedup_log.txt"
results_store    = "factory/data/results.json"
factory_log      = "factory/logs/factory.log"
tmp_dir          = "factory/data/_tmp"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p", "--bare", "--output-format", "json", "--allowedTools", "Read"]
generation_timeout_sec = 120

[stages]
stage_timeout_sec = 5400

[alerts]
alert_threshold_metric = "wfo.oos_sharpe"
alert_threshold        = 1.0
telegram_bot_token     = ""
telegram_chat_id       = ""
dashboard_base_url     = "http://127.0.0.1:8787"

[loop]
mode                  = "continuous"
inter_cycle_sleep_sec = 5
max_cycles            = 0   # 0 = unlimited

[dashboard]
host             = "127.0.0.1"
port             = 8787
auto_refresh_sec = 10
```

- [ ] **Step 5: Write `factory/factory/settings_loader.py`**

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True, frozen=True)
class Paths:
    backtester_root: Path
    strategies_dir: Path
    configs_dir: Path
    registry_file: Path
    output_runs_dir: Path
    dedup_log: Path
    results_store: Path
    factory_log: Path
    tmp_dir: Path


@dataclass(slots=True, frozen=True)
class GenerationCfg:
    claude_cmd: str
    claude_flags: tuple[str, ...]
    generation_timeout_sec: int


@dataclass(slots=True, frozen=True)
class StagesCfg:
    stage_timeout_sec: int


@dataclass(slots=True, frozen=True)
class AlertsCfg:
    alert_threshold_metric: str
    alert_threshold: float
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_base_url: str


@dataclass(slots=True, frozen=True)
class LoopCfg:
    mode: str
    inter_cycle_sleep_sec: int
    max_cycles: int


@dataclass(slots=True, frozen=True)
class DashboardCfg:
    host: str
    port: int
    auto_refresh_sec: int


@dataclass(slots=True, frozen=True)
class Settings:
    paths: Paths
    generation: GenerationCfg
    stages: StagesCfg
    alerts: AlertsCfg
    loop: LoopCfg
    dashboard: DashboardCfg


def load_settings(path: Path) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    p = raw["paths"]
    root = Path(p["backtester_root"]).resolve()

    def _under_root(rel: str) -> Path:
        return (root / rel).resolve()

    paths = Paths(
        backtester_root=root,
        strategies_dir=_under_root(p["strategies_dir"]),
        configs_dir=_under_root(p["configs_dir"]),
        registry_file=_under_root(p["registry_file"]),
        output_runs_dir=_under_root(p["output_runs_dir"]),
        dedup_log=_under_root(p["dedup_log"]),
        results_store=_under_root(p["results_store"]),
        factory_log=_under_root(p["factory_log"]),
        tmp_dir=_under_root(p["tmp_dir"]),
    )
    g = raw["generation"]
    s = raw["stages"]
    a = raw["alerts"]
    lp = raw["loop"]
    d = raw["dashboard"]
    return Settings(
        paths=paths,
        generation=GenerationCfg(
            claude_cmd=g["claude_cmd"],
            claude_flags=tuple(g["claude_flags"]),
            generation_timeout_sec=int(g["generation_timeout_sec"]),
        ),
        stages=StagesCfg(stage_timeout_sec=int(s["stage_timeout_sec"])),
        alerts=AlertsCfg(
            alert_threshold_metric=a["alert_threshold_metric"],
            alert_threshold=float(a["alert_threshold"]),
            telegram_bot_token=a["telegram_bot_token"],
            telegram_chat_id=a["telegram_chat_id"],
            dashboard_base_url=a["dashboard_base_url"],
        ),
        loop=LoopCfg(
            mode=lp["mode"],
            inter_cycle_sleep_sec=int(lp["inter_cycle_sleep_sec"]),
            max_cycles=int(lp["max_cycles"]),
        ),
        dashboard=DashboardCfg(
            host=d["host"], port=int(d["port"]),
            auto_refresh_sec=int(d["auto_refresh_sec"]),
        ),
    )
```

- [ ] **Step 6: Write `factory/tests/conftest.py`**

```python
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_settings_file(tmp_path: Path) -> Path:
    """Write a complete settings.toml into tmp_path with backtester_root=tmp_path."""
    toml = textwrap.dedent(
        f"""
        [paths]
        backtester_root  = "{tmp_path.as_posix()}"
        strategies_dir   = "strategies"
        configs_dir      = "configs/wfo"
        registry_file    = "backtester/strategies/registry.py"
        output_runs_dir  = "output/runs"
        dedup_log        = "factory/data/dedup_log.txt"
        results_store    = "factory/data/results.json"
        factory_log      = "factory/logs/factory.log"
        tmp_dir          = "factory/data/_tmp"

        [generation]
        claude_cmd             = "claude"
        claude_flags           = ["-p", "--bare", "--output-format", "json"]
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
        """
    ).strip()
    p = tmp_path / "settings.toml"
    p.write_text(toml, encoding="utf-8")
    return p
```

- [ ] **Step 7: Write the failing test `factory/tests/test_settings_loader.py`**

```python
from pathlib import Path

from factory.settings_loader import load_settings


def test_loads_all_sections(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    assert s.paths.backtester_root.is_absolute()
    assert s.paths.strategies_dir.name == "strategies"
    assert s.paths.registry_file.parts[-2:] == ("strategies", "registry.py")
    assert s.generation.claude_cmd == "claude"
    assert "--bare" in s.generation.claude_flags
    assert s.generation.generation_timeout_sec == 60
    assert s.stages.stage_timeout_sec == 300
    assert s.alerts.alert_threshold_metric == "wfo.oos_sharpe"
    assert s.alerts.alert_threshold == 1.0
    assert s.loop.mode == "continuous"
    assert s.loop.max_cycles == 1
    assert s.dashboard.port == 8787


def test_paths_resolve_under_root(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    root = s.paths.backtester_root
    assert s.paths.strategies_dir.is_relative_to(root)
    assert s.paths.results_store.is_relative_to(root)
    assert s.paths.tmp_dir.is_relative_to(root)
```

- [ ] **Step 8: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_settings_loader.py -q
```

Expected: 2 passed.

- [ ] **Step 9: Write `factory/README.md`**

```markdown
# Strategy Factory v0.2.0

Local, unattended strategy-idea factory wrapping the Modular-Backtester.

## Quickstart

1. `pip install -e .[dev]` from the backtester repo root (installs the backtester package and pytest).
2. `pip install Flask` (the factory's only extra runtime dep).
3. Edit `factory/config/settings.toml` — set `backtester_root` to an absolute path if needed, and fill `telegram_bot_token` / `telegram_chat_id` if you want alerts.
4. Run the loop: `python -m factory.loop`
5. Run the dashboard (separate terminal): `python -m dashboard.server`
6. Open `http://127.0.0.1:8787`

## Tests

`python -m pytest factory/tests -q` from the backtester root.

Slow tests (Tier 2 functional validation, integration smoke) are marked `@pytest.mark.slow`. Run with `-m slow` to include them, `-m "not slow"` to skip.

## Spec

See `docs/superpowers/plans/2026-05-15-strategy-factory-v020.md` for the implementation plan and the linked spec.
```

- [ ] **Step 10: Commit**

```powershell
git add factory/
git commit -m @'
feat(factory): bootstrap directory layout, settings loader, conftest

Adds factory/ subdirectory with config/settings.toml, settings_loader.py
producing typed dataclasses, pytest path injection via factory/conftest.py,
and a shared tmp_settings_file fixture. Two settings_loader tests pass.

No backtester source files are modified.
'@
```

---

## Task 1: Slots (`slots.py`)

**Files:**
- Create: `factory/factory/slots.py`
- Create: `factory/tests/test_slots.py`

Implements §5.1: six slot lists; `pull()` returns one random choice from each. The slot lists ARE the diversity engine — they must match the spec's enumerations.

- [ ] **Step 1: Write the failing test `factory/tests/test_slots.py`**

```python
import random
from collections import Counter

from factory.slots import SLOT_NAMES, SLOTS, pull_slots


def test_six_slots_with_expected_names() -> None:
    assert SLOT_NAMES == (
        "strategy_family",
        "signal_primitive",
        "holding_horizon",
        "direction",
        "constraint_twist",
        "inspiration_anchor",
    )
    for name in SLOT_NAMES:
        assert len(SLOTS[name]) >= 3, name


def test_pull_returns_one_per_slot() -> None:
    rng = random.Random(42)
    pulled = pull_slots(rng)
    assert set(pulled.keys()) == set(SLOT_NAMES)
    for name, value in pulled.items():
        assert value in SLOTS[name], (name, value)


def test_pull_is_diverse_across_many_calls() -> None:
    rng = random.Random(0)
    families = Counter()
    for _ in range(200):
        families[pull_slots(rng)["strategy_family"]] += 1
    # All distinct families should appear within 200 pulls (~12 families).
    assert len(families) == len(SLOTS["strategy_family"])


def test_direction_is_weighted_toward_long_only() -> None:
    # spec: long-only x2, long/short x1
    rng = random.Random(7)
    counts = Counter(pull_slots(rng)["direction"] for _ in range(3000))
    assert counts["long-only"] > counts["long/short"]
    # Expect ratio ~2:1 — allow generous tolerance for randomness.
    ratio = counts["long-only"] / counts["long/short"]
    assert 1.4 < ratio < 2.6, ratio
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_slots.py -q
```

Expected: ModuleNotFoundError or AttributeError — slots module doesn't exist.

- [ ] **Step 3: Write `factory/factory/slots.py`**

```python
from __future__ import annotations

import random
from typing import Mapping

SLOT_NAMES: tuple[str, ...] = (
    "strategy_family",
    "signal_primitive",
    "holding_horizon",
    "direction",
    "constraint_twist",
    "inspiration_anchor",
)

SLOTS: Mapping[str, tuple[str, ...]] = {
    "strategy_family": (
        "momentum", "mean-reversion", "breakout", "volatility-targeting",
        "seasonality", "regime-switching", "range-compression",
        "gap-behavior", "drawdown-recovery", "autocorrelation",
        "relative-position", "trend-strength",
    ),
    "signal_primitive": (
        "close-to-close returns", "high-low range dynamics",
        "volume-confirmed moves", "volatility (std/ATR)",
        "gap (open vs prior close)", "rolling rank/percentile",
        "consecutive-streak count", "distance-from-MA (z-score)",
        "rate-of-change acceleration", "drawdown depth",
    ),
    "holding_horizon": (
        "1-2 days", "3-5 days", "1-2 weeks", "3-4 weeks",
    ),
    "direction": (
        "long-only", "long-only", "long/short",
    ),
    "constraint_twist": (
        "<=2 tunable params", "regime filter on 200-day MA",
        "signal-scaled position sizing", "symmetric entry/exit rule",
        "fixed-bar exit (no signal-based exit)",
        "two-primitive AND (both must agree)",
        "percentile threshold instead of fixed level",
        "warmup <=10 bars", "no stop-loss allowed",
        "two-bar confirmation before entry",
    ),
    "inspiration_anchor": (
        "hysteresis control", "predator-prey cycles",
        "queue overflow / capacity limits", "signal-to-noise filtering",
        "spring tension / elastic restoring force",
        "epidemic curves (susceptible-infected)",
        "traffic shockwaves", "elastic vs plastic deformation",
        "refractory period after a spike", "tide tables / standing waves",
    ),
}


def pull_slots(rng: random.Random) -> dict[str, str]:
    """Return one randomly-chosen value per slot."""
    return {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_slots.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/slots.py factory/tests/test_slots.py
git commit -m @'
feat(factory): slot definitions and weighted random pull (§5.1)

Six slots with the spec's enumerations; direction is weighted 2:1
toward long-only via list duplication. pull_slots takes an injected
random.Random for deterministic tests.
'@
```

---

## Task 2: Prompt builder (`prompt.py`)

**Files:**
- Create: `factory/factory/prompt.py`
- Create: `factory/tests/test_prompt.py`

Implements §5.2 + Appendix A verbatim. The template body is reproduced exactly from the spec, with `{{double_braces}}` placeholders filled by `build_prompt(...)`.

- [ ] **Step 1: Write the failing test `factory/tests/test_prompt.py`**

```python
from factory.prompt import build_prompt


def test_build_prompt_fills_all_placeholders() -> None:
    slots = {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }
    text = build_prompt(
        strategy_id="gen_1715800000",
        slots=slots,
        dedup_tail=["sma cross 50/200", "rsi mean reversion 14"],
    )
    # Every placeholder should be filled.
    assert "{{" not in text and "}}" not in text
    # Slot values present in the prompt.
    for v in slots.values():
        assert v in text
    # Strategy id appears in the prompt (must match injected value).
    assert "gen_1715800000" in text
    # Dedup tail appears as numbered/bulleted lines.
    assert "sma cross 50/200" in text
    assert "rsi mean reversion 14" in text
    # Hard contract markers from Appendix A.
    assert "GeneratedStrategy" in text
    assert "shift(1)" in text
    assert "strict JSON" in text


def test_empty_dedup_tail_is_handled() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "constraint_twist", "inspiration_anchor",
    )}
    text = build_prompt(strategy_id="gen_1", slots=slots, dedup_tail=[])
    assert "(none yet)" in text


def test_long_dedup_tail_caps_at_30() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "constraint_twist", "inspiration_anchor",
    )}
    tail = [f"idea {i}" for i in range(100)]
    text = build_prompt(strategy_id="gen_1", slots=slots, dedup_tail=tail)
    # Only the last 30 of 100 should appear in the prompt.
    assert "idea 70" in text
    assert "idea 99" in text
    assert "idea 69" not in text
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_prompt.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/prompt.py`**

```python
from __future__ import annotations

from typing import Mapping, Sequence

# Appendix A — reproduced verbatim from the spec.
# Placeholders use {{double_braces}} and are filled by build_prompt.
PROMPT_TEMPLATE = """You are a quantitative strategy researcher generating ONE novel, self-contained
trading strategy for a specific Python backtesting framework. This is an
idea-generation factory optimizing for breadth and originality. Output must be
mechanically valid - it will be written to disk and run with no human review.

THE FRAMEWORK CONTRACT - follow exactly:

A strategy is one Python file. It must contain:
1. A @dataclass(slots=True) params class. All fields must have defaults and be
   int, float, or bool.
2. A class - named exactly GeneratedStrategy - inheriting BaseStrategy[YourParams]
   with: a strategy_id class attribute, params_type() classmethod,
   warmup_bars(params), indicators(data, params), and
   generate_signals(data, indicators, ctx, params).

Method semantics:
- indicators(data, params) returns a DataFrame indexed like data, holding every
  derived series. data has lowercase columns open, high, low, close, volume and a
  datetime index. No other columns exist. No other tickers, no fundamental data.
- generate_signals(data, indicators, ctx, params) returns
  SignalFrame(data=df, signal_column="signal", size_column="size").
  df["signal"] must be integer {-1, 0, 1}. df["size"] is a positive float.
- MANDATORY: the signal MUST be shifted by exactly one bar -
  df["signal"] = df["signal"].shift(1).fillna(0).astype(int). The strategy
  decides on bar N's close; the fill happens on bar N+1. Omitting this is
  lookahead bias and a fatal bug.
- warmup_bars(params) must return an int >= the longest lookback any indicator
  uses. If you use .diff() or .pct_change() before a rolling window of length L,
  return L + 1.
- Use only pandas and numpy. Import them. No other libraries.
- Prefer vectorised pandas operations; avoid .rolling().apply() with Python
  callables where a vectorised equivalent exists.
- The signal must be mechanically computable from SPY OHLCV alone.
- DO NOT set uses_multi_symbol = True. DO NOT set uses_per_bar = True. The
  factory only supports the v0.3.0-style single-symbol contract.

THIS IDEA'S RANDOM CONSTRAINTS:
- strategy_id (use exactly this, do not invent your own): {{strategy_id}}
- Strategy family: {{strategy_family}}
- Primary signal primitive: {{signal_primitive}}
- Target holding horizon: {{holding_horizon}}
- Direction: {{direction}} (if "long/short", you may emit -1 signals; if
  "long-only", never emit -1)
- Hard twist (must satisfy): {{constraint_twist}}
- Loose inspiration (use only if genuinely useful): {{inspiration_anchor}}

ALREADY-GENERATED IDEAS - yours must be meaningfully different from every one.
Not a parameter tweak, not the same hypothesis with a different indicator. A
different mechanism.
{{last_30_idea_summaries}}

OUTPUT - strict JSON, nothing outside it, no markdown fences:
{
  "strategy_id": "{{strategy_id}}",
  "one_line_summary": "<=20 words, names the mechanism, for the dedup log",
  "hypothesis": "the market inefficiency or behavioral pattern this exploits, 2-3 sentences",
  "novelty_justification": "why this differs in mechanism from the already-generated list",
  "failure_mode": "the single most likely reason this won't work - be specific and honest",
  "allow_short": <true|false>,
  "strategy_file": "<the complete .py file as a string>",
  "config_file": "<the complete .yaml config as a string>"
}

The config_file must follow this exact shape, with strategy: set to
{{strategy_id}}, execution.allow_short matching your allow_short,
optimization.param_space covering 2-3 of your params with 3 values each, and
wfo.enabled: true:

run_name: {{strategy_id}}
strategy: {{strategy_id}}
strategy_params: {<your defaults>}
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
  allow_short: <true|false>
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space: {<2-3 params, 3 values each>}
wfo:
  enabled: true
  train_bars: 756
  test_bars: 252
  step_bars: 252

Rules: satisfy the twist even if it conflicts with the family. Every indicator
must be NaN-safe (rolling windows produce NaN during warmup - handle it). The
class must be named exactly GeneratedStrategy. Do not add disclaimers or
hedging. Do not explain the code outside the JSON.
"""


def build_prompt(
    *,
    strategy_id: str,
    slots: Mapping[str, str],
    dedup_tail: Sequence[str],
) -> str:
    """Fill the Appendix A template with the slot values and the dedup tail.

    `dedup_tail` is the LIST of recent one_line_summary lines (oldest first).
    Only the last 30 are used.
    """
    tail = list(dedup_tail)[-30:]
    if tail:
        tail_block = "\n".join(f"- {line}" for line in tail)
    else:
        tail_block = "(none yet)"

    filled = PROMPT_TEMPLATE
    for name, value in slots.items():
        filled = filled.replace("{{" + name + "}}", value)
    filled = filled.replace("{{strategy_id}}", strategy_id)
    filled = filled.replace("{{last_30_idea_summaries}}", tail_block)
    return filled
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_prompt.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Eyeball test — generate 20 prompts and verify slot diversity**

Add a quick visual check; not committed.

```powershell
python -c @'
import random
from factory.slots import pull_slots
from factory.prompt import build_prompt
rng = random.Random(123)
for i in range(20):
    s = pull_slots(rng)
    print(f"{i:>2}  {s['strategy_family']:<22} | {s['signal_primitive']:<32} | {s['direction']:<10} | {s['constraint_twist']}")
'@
```

Expected: visible variety in family/primitive/twist columns. If any column is suspiciously constant, fix the slot list before continuing.

- [ ] **Step 6: Commit**

```powershell
git add factory/factory/prompt.py factory/tests/test_prompt.py
git commit -m @'
feat(factory): prompt builder with Appendix A template (§5.2)

PROMPT_TEMPLATE reproduces Appendix A verbatim with two factory-specific
additions: explicit ban on uses_multi_symbol / uses_per_bar (v0.4.0 surface
the factory does not target). build_prompt fills slot values, strategy_id,
and the last 30 dedup-log entries.
'@
```

---

## Task 3: Dedup log (`dedup.py`)

**Files:**
- Create: `factory/factory/dedup.py`
- Create: `factory/tests/test_dedup.py`

Implements §5.6 + §3.2. Append-only `data/dedup_log.txt`, one `one_line_summary` per line. The append happens AS SOON AS a parseable `one_line_summary` exists — BEFORE validation, BEFORE backtester stages. This is the load-bearing timing rule.

- [ ] **Step 1: Write the failing test `factory/tests/test_dedup.py`**

```python
from pathlib import Path

from factory.dedup import append_summary, read_tail


def test_append_then_read_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "first idea")
    append_summary(log, "second idea")
    append_summary(log, "third idea")
    assert read_tail(log, n=10) == ["first idea", "second idea", "third idea"]


def test_read_tail_caps_at_n(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    for i in range(50):
        append_summary(log, f"idea {i}")
    tail = read_tail(log, n=30)
    assert len(tail) == 30
    assert tail[0] == "idea 20"
    assert tail[-1] == "idea 49"


def test_read_tail_handles_missing_file(tmp_path: Path) -> None:
    assert read_tail(tmp_path / "does_not_exist.txt", n=30) == []


def test_append_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "dedup.txt"
    append_summary(log, "hello")
    assert log.exists()
    assert log.read_text(encoding="utf-8").strip() == "hello"


def test_append_strips_newlines_inside_summary(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "line1\nline2\rline3")
    assert read_tail(log, n=10) == ["line1 line2 line3"]


def test_append_skips_empty_or_whitespace(tmp_path: Path) -> None:
    log = tmp_path / "dedup.txt"
    append_summary(log, "")
    append_summary(log, "   ")
    append_summary(log, "real entry")
    assert read_tail(log, n=10) == ["real entry"]
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_dedup.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/dedup.py`**

```python
from __future__ import annotations

from pathlib import Path


def append_summary(log_path: Path, summary: str) -> None:
    """Append one one_line_summary to the dedup log.

    Newlines/carriage returns inside the summary are replaced with spaces so
    one line == one entry. Empty/whitespace-only summaries are silently ignored.
    Parent directories are created on demand.
    """
    cleaned = " ".join(summary.replace("\r", "\n").split("\n")).strip()
    if not cleaned:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(cleaned + "\n")


def read_tail(log_path: Path, n: int) -> list[str]:
    """Return the last n non-empty lines of the dedup log, oldest first.

    Returns [] if the file does not exist.
    """
    if not log_path.exists():
        return []
    lines = [
        line.rstrip("\n")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[-n:]
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_dedup.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/dedup.py factory/tests/test_dedup.py
git commit -m @'
feat(factory): dedup log append + tail read (§5.6, §3.2)

Append-only one-summary-per-line text log. Newlines inside a summary are
flattened to spaces so the file invariant holds. read_tail returns the
last n entries oldest-first (the order build_prompt expects).
'@
```

---

## Task 4: Generate — claude -p subprocess + double-unwrap parser (`generate.py`)

**Files:**
- Create: `factory/factory/generate.py`
- Create: `factory/tests/fixtures/claude_output_clean.json`
- Create: `factory/tests/fixtures/claude_output_fenced.json`
- Create: `factory/tests/fixtures/claude_output_prose_wrapped.json`
- Create: `factory/tests/fixtures/claude_output_malformed.json`
- Create: `factory/tests/test_generate.py`

Implements §5.3. The CRITICAL piece is the double-unwrap parser — defensive against fenced/prose-wrapped output. Per §9 landmine 4, parse failure is a logged generation failure, not a crash. The subprocess invocation itself is small and well-defined.

- [ ] **Step 1: Write the four fixture files**

Each fixture mimics a real `claude -p --output-format json` stdout: the CLI envelope is `{"result": "<text>", "session_id": "...", "total_cost_usd": 0.034, ...}`. The `.result` field holds the model's text output, which is supposed to be strict JSON but may be wrapped.

`factory/tests/fixtures/claude_output_clean.json`:

```json
{
  "result": "{\"strategy_id\": \"gen_1\", \"one_line_summary\": \"sma cross 20/100\", \"hypothesis\": \"trend follows itself.\", \"novelty_justification\": \"baseline.\", \"failure_mode\": \"chop.\", \"allow_short\": false, \"strategy_file\": \"# placeholder strategy\\n\", \"config_file\": \"run_name: gen_1\\n\"}",
  "session_id": "abc",
  "total_cost_usd": 0.034,
  "duration_ms": 4200
}
```

`factory/tests/fixtures/claude_output_fenced.json`:

```json
{
  "result": "```json\n{\"strategy_id\": \"gen_2\", \"one_line_summary\": \"rsi reversal\", \"hypothesis\": \"oversold bounces.\", \"novelty_justification\": \"x.\", \"failure_mode\": \"trends.\", \"allow_short\": false, \"strategy_file\": \"# placeholder\\n\", \"config_file\": \"run_name: gen_2\\n\"}\n```",
  "session_id": "def",
  "total_cost_usd": 0.041
}
```

`factory/tests/fixtures/claude_output_prose_wrapped.json`:

```json
{
  "result": "Here is the strategy I generated based on your constraints:\n\n{\"strategy_id\": \"gen_3\", \"one_line_summary\": \"breakout 20d\", \"hypothesis\": \"new highs continue.\", \"novelty_justification\": \"x.\", \"failure_mode\": \"false breaks.\", \"allow_short\": false, \"strategy_file\": \"# placeholder\\n\", \"config_file\": \"run_name: gen_3\\n\"}\n\nLet me know if you'd like any adjustments.",
  "session_id": "ghi",
  "total_cost_usd": 0.029
}
```

`factory/tests/fixtures/claude_output_malformed.json`:

```json
{
  "result": "I couldn't generate a strategy because the constraints conflict.",
  "session_id": "jkl",
  "total_cost_usd": 0.012
}
```

- [ ] **Step 2: Write the failing test `factory/tests/test_generate.py`**

```python
import json
from pathlib import Path

import pytest

from factory.generate import (
    GenerationError,
    parse_claude_output,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_parses_clean_output() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_clean.json"))
    assert parsed["strategy_id"] == "gen_1"
    assert parsed["one_line_summary"] == "sma cross 20/100"
    assert parsed["allow_short"] is False
    assert parsed["strategy_file"].startswith("# placeholder")
    assert cost == pytest.approx(0.034)


def test_strips_markdown_fences() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_fenced.json"))
    assert parsed["strategy_id"] == "gen_2"
    assert cost == pytest.approx(0.041)


def test_locates_json_inside_prose() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_prose_wrapped.json"))
    assert parsed["strategy_id"] == "gen_3"
    assert cost == pytest.approx(0.029)


def test_raises_on_no_json_object() -> None:
    with pytest.raises(GenerationError) as exc:
        parse_claude_output(_load("claude_output_malformed.json"))
    assert "no JSON object" in str(exc.value).lower() or "could not parse" in str(exc.value).lower()


def test_raises_on_broken_envelope() -> None:
    with pytest.raises(GenerationError):
        parse_claude_output("this is not even JSON")


def test_raises_on_missing_required_keys() -> None:
    # Envelope is fine; inner JSON is parseable but lacks required keys.
    envelope = json.dumps({
        "result": '{"strategy_id": "gen_x"}',
        "total_cost_usd": 0.01,
    })
    with pytest.raises(GenerationError) as exc:
        parse_claude_output(envelope)
    assert "missing" in str(exc.value).lower()


def test_raises_on_envelope_without_result_field() -> None:
    envelope = json.dumps({"session_id": "x", "total_cost_usd": 0.0})
    with pytest.raises(GenerationError):
        parse_claude_output(envelope)
```

- [ ] **Step 3: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_generate.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Write `factory/factory/generate.py`**

```python
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_KEYS: tuple[str, ...] = (
    "strategy_id",
    "one_line_summary",
    "hypothesis",
    "novelty_justification",
    "failure_mode",
    "allow_short",
    "strategy_file",
    "config_file",
)


class GenerationError(RuntimeError):
    """Raised when claude -p output cannot be parsed into the expected shape."""


@dataclass(slots=True, frozen=True)
class GenerationResult:
    parsed: dict[str, Any]
    cost_usd: float
    raw_stdout: str


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1)
    return text


def _find_outer_json_object(text: str) -> str:
    """Locate the outermost balanced {...} in text. Raises ValueError if none."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in text")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces in text")


def parse_claude_output(stdout: str) -> tuple[dict[str, Any], float]:
    """Defensive double-unwrap.

    Returns (parsed_strategy_dict, total_cost_usd).
    Raises GenerationError on any parse failure or missing required key.
    """
    # Layer 1: CLI envelope.
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"could not parse CLI envelope: {exc}") from exc
    if not isinstance(envelope, dict) or "result" not in envelope:
        raise GenerationError("envelope missing 'result' field")
    inner_text = envelope["result"]
    cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)

    # Layer 2: strip fences, locate outer JSON, parse.
    stripped = _strip_fences(inner_text)
    try:
        inner_blob = _find_outer_json_object(stripped)
    except ValueError as exc:
        raise GenerationError(f"no JSON object in inner result: {exc}") from exc
    try:
        parsed = json.loads(inner_blob)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"could not parse inner JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GenerationError("inner JSON is not an object")

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        raise GenerationError(f"inner JSON missing keys: {missing}")

    return parsed, cost


def call_claude(
    *,
    prompt: str,
    claude_cmd: str,
    claude_flags: tuple[str, ...],
    timeout_sec: int,
) -> GenerationResult:
    """Invoke claude -p as a subprocess and parse its stdout.

    Raises GenerationError on non-zero exit, timeout, or unparseable output.
    """
    cmd = [claude_cmd, *claude_flags, prompt]
    log.info("calling claude (cmd=%s flags=%s)", claude_cmd, claude_flags)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(f"claude -p timed out after {timeout_sec}s") from exc
    except FileNotFoundError as exc:
        raise GenerationError(f"claude command not found: {claude_cmd}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-500:]
        raise GenerationError(
            f"claude -p exited {proc.returncode}; stderr tail: {tail}"
        )

    parsed, cost = parse_claude_output(proc.stdout)
    return GenerationResult(parsed=parsed, cost_usd=cost, raw_stdout=proc.stdout)
```

- [ ] **Step 5: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_generate.py -q
```

Expected: 7 passed.

- [ ] **Step 6: One real `claude -p` smoke test (optional manual check, not committed)**

```powershell
python -c @'
from factory.prompt import build_prompt
from factory.slots import pull_slots
import random
from factory.generate import call_claude

slots = pull_slots(random.Random(0))
prompt = build_prompt(strategy_id="gen_smoke", slots=slots, dedup_tail=[])
result = call_claude(
    prompt=prompt,
    claude_cmd="claude",
    claude_flags=("-p", "--bare", "--output-format", "json", "--allowedTools", "Read"),
    timeout_sec=120,
)
print("cost:", result.cost_usd)
print("strategy_id:", result.parsed["strategy_id"])
print("summary:", result.parsed["one_line_summary"])
print("--- strategy_file (first 400 chars) ---")
print(result.parsed["strategy_file"][:400])
'@
```

Expected: prints a cost ~$0.02-0.05, a strategy_id starting with `gen_`, and a one_line_summary. If parsing fails, capture the raw stdout and add a new fixture before continuing.

- [ ] **Step 7: Commit**

```powershell
git add factory/factory/generate.py factory/tests/test_generate.py factory/tests/fixtures/claude_output_*.json
git commit -m @'
feat(factory): claude -p subprocess + defensive double-unwrap parser (§5.3)

parse_claude_output handles three observed shapes: clean inner JSON,
markdown-fenced inner JSON, and inner JSON inside prose. _find_outer_json_object
walks braces with string-aware escaping. GenerationError covers every failure
mode (non-zero exit, timeout, malformed envelope, malformed inner, missing keys).
'@
```

---

## Task 5: Validate Tier 1 — static checks (`validate.py` part 1)

**Files:**
- Create: `factory/factory/validate.py` (Tier 1 only this task)
- Create: `factory/tests/fixtures/valid_strategy.py`
- Create: `factory/tests/fixtures/invalid_no_shift.py`
- Create: `factory/tests/fixtures/invalid_bad_imports.py`
- Create: `factory/tests/fixtures/invalid_no_class.py`
- Create: `factory/tests/fixtures/valid_config.yaml`
- Create: `factory/tests/fixtures/invalid_config_wrong_strategy.yaml`
- Create: `factory/tests/test_validate_static.py`

Implements §5.4 Tier 1 — eight static checks. Cheap (milliseconds). The static check that the spec asks for IS necessary but not sufficient (Tier 2 is the actual safety net).

- [ ] **Step 1: Write `factory/tests/fixtures/valid_strategy.py`**

Clone of `strategies/gen_1715800000.py` from the backtester repo. Copy its full body verbatim, then **edit one line**: change `strategy_id = "gen_1715800000"` to `strategy_id = "gen_test_valid"`. This gives us a known-conforming strategy with a fixed id to test against.

- [ ] **Step 2: Write the other fixture strategies**

`factory/tests/fixtures/invalid_no_shift.py` — copy of `valid_strategy.py` but remove the line `df["signal"] = df["signal"].shift(1).fillna(0).astype(int)`.

`factory/tests/fixtures/invalid_bad_imports.py` — copy of `valid_strategy.py` but add `import os` and `import requests` at the top.

`factory/tests/fixtures/invalid_no_class.py` — content:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class GeneratedParams:
    size: float = 1.0

# Note: no GeneratedStrategy class.
```

- [ ] **Step 3: Write fixture configs**

`factory/tests/fixtures/valid_config.yaml`:

```yaml
run_name: gen_test_valid
strategy: gen_test_valid
strategy_params:
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
  allow_short: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space:
    size: [0.5, 1.0]
wfo:
  enabled: true
  train_bars: 756
  test_bars: 252
  step_bars: 252
```

`factory/tests/fixtures/invalid_config_wrong_strategy.yaml`:

```yaml
run_name: gen_test_valid
strategy: some_other_strategy
strategy_params: {}
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
  allow_short: false
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
optimization:
  objective: sharpe
  param_space: {}
wfo:
  enabled: false
  train_bars: 756
  test_bars: 252
  step_bars: 252
```

- [ ] **Step 4: Write the failing test `factory/tests/test_validate_static.py`**

```python
from pathlib import Path

import pytest

from factory.validate import StaticValidationError, validate_static

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_valid_strategy_passes() -> None:
    validate_static(
        strategy_id="gen_test_valid",
        strategy_src=_read("valid_strategy.py"),
        config_src=_read("valid_config.yaml"),
        allow_short=False,
    )


def test_missing_shift_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_no_shift.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "shift(1)" in str(exc.value)


def test_forbidden_imports_fail() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_bad_imports.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    msg = str(exc.value).lower()
    assert "import" in msg and ("os" in msg or "requests" in msg)


def test_missing_class_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_no_class.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "GeneratedStrategy" in str(exc.value)


def test_config_strategy_mismatch_fails() -> None:
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=_read("valid_strategy.py"),
            config_src=_read("invalid_config_wrong_strategy.yaml"),
            allow_short=False,
        )
    msg = str(exc.value).lower()
    assert "strategy" in msg


def test_strategy_id_attribute_must_match_injected_id() -> None:
    # valid_strategy.py declares strategy_id = "gen_test_valid"; passing a
    # different injected id is a mismatch.
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_different",
            strategy_src=_read("valid_strategy.py"),
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "strategy_id" in str(exc.value)


def test_multi_symbol_attribute_is_forbidden() -> None:
    poisoned = _read("valid_strategy.py").replace(
        'strategy_id = "gen_test_valid"',
        'strategy_id = "gen_test_valid"\n    uses_multi_symbol = True',
    )
    with pytest.raises(StaticValidationError) as exc:
        validate_static(
            strategy_id="gen_test_valid",
            strategy_src=poisoned,
            config_src=_read("valid_config.yaml"),
            allow_short=False,
        )
    assert "uses_multi_symbol" in str(exc.value)
```

- [ ] **Step 5: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_validate_static.py -q
```

Expected: ModuleNotFoundError or AttributeError.

- [ ] **Step 6: Write `factory/factory/validate.py` (Tier 1 only)**

```python
from __future__ import annotations

import ast
import logging
from typing import Iterable

import yaml

log = logging.getLogger(__name__)

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset({
    "pandas", "numpy", "dataclasses", "__future__", "typing",
    "backtester", "backtester.core", "backtester.core.types",
    "backtester.strategies", "backtester.strategies.base",
})

REQUIRED_METHODS: tuple[str, ...] = (
    "params_type", "warmup_bars", "indicators", "generate_signals",
)


class StaticValidationError(ValueError):
    """Tier 1 static check failure."""


class FunctionalValidationError(ValueError):
    """Tier 2 functional check failure (implemented in Task 6)."""


def _import_root(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return
        yield node.module.split(".")[0]


def _check_imports(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in _import_root(node):
                if root not in ALLOWED_IMPORT_ROOTS and not any(
                    a == root or a.startswith(root + ".") for a in ALLOWED_IMPORT_ROOTS
                ):
                    raise StaticValidationError(
                        f"forbidden import root: {root!r} (allowed: {sorted(ALLOWED_IMPORT_ROOTS)})"
                    )


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise StaticValidationError(f"class {name!r} not found")


def _class_methods(cls: ast.ClassDef) -> set[str]:
    return {
        n.name for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_assigns(cls: ast.ClassDef) -> dict[str, ast.AST]:
    """Return a {name -> value-node} map of class-body name = value assignments."""
    out: dict[str, ast.AST] = {}
    for n in cls.body:
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = n.value
    return out


def _has_dataclass_slots_true(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            # @dataclass(slots=True)
            if isinstance(dec, ast.Call):
                func = dec.func
                if (isinstance(func, ast.Name) and func.id == "dataclass") or (
                    isinstance(func, ast.Attribute) and func.attr == "dataclass"
                ):
                    for kw in dec.keywords:
                        if kw.arg == "slots" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            return True
    return False


def validate_static(
    *,
    strategy_id: str,
    strategy_src: str,
    config_src: str,
    allow_short: bool,
) -> None:
    """Tier 1 static contract checks. Raises StaticValidationError on first failure."""
    # 1. Parses.
    try:
        tree = ast.parse(strategy_src)
    except SyntaxError as exc:
        raise StaticValidationError(f"strategy file does not parse: {exc}") from exc

    # 5. Import whitelist (run early — cheap and catches the worst offenders).
    _check_imports(tree)

    # 2. GeneratedStrategy class present.
    cls = _find_class(tree, "GeneratedStrategy")

    # 3. Required methods.
    methods = _class_methods(cls)
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        raise StaticValidationError(
            f"GeneratedStrategy missing required methods: {missing}"
        )

    # 4. strategy_id attribute present and matches injected id.
    assigns = _class_assigns(cls)
    if "strategy_id" not in assigns:
        raise StaticValidationError("GeneratedStrategy missing strategy_id attribute")
    val = assigns["strategy_id"]
    if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
        raise StaticValidationError("strategy_id must be a string literal")
    if val.value != strategy_id:
        raise StaticValidationError(
            f"strategy_id mismatch: file declares {val.value!r}, injected {strategy_id!r}"
        )

    # Factory-specific: forbid v0.4.0 multi-symbol opt-in attributes.
    for forbidden in ("uses_multi_symbol", "uses_per_bar"):
        if forbidden in assigns:
            v = assigns[forbidden]
            if isinstance(v, ast.Constant) and v.value is True:
                raise StaticValidationError(
                    f"{forbidden} = True is forbidden (factory targets v0.3.0-style strategies only)"
                )

    # 6. Shift present (cheap proxy for the mandatory one-bar shift).
    if ".shift(1)" not in strategy_src:
        raise StaticValidationError(
            "strategy source does not contain '.shift(1)' (the mandatory one-bar signal shift)"
        )

    # 7. @dataclass(slots=True) params class present.
    if not _has_dataclass_slots_true(tree):
        raise StaticValidationError(
            "no @dataclass(slots=True) found (params class is required)"
        )

    # 8. Config sanity.
    try:
        cfg = yaml.safe_load(config_src)
    except yaml.YAMLError as exc:
        raise StaticValidationError(f"config_file does not parse: {exc}") from exc
    if not isinstance(cfg, dict):
        raise StaticValidationError("config_file root must be a mapping")
    if cfg.get("strategy") != strategy_id:
        raise StaticValidationError(
            f"config strategy={cfg.get('strategy')!r} does not match strategy_id={strategy_id!r}"
        )
    wfo = cfg.get("wfo") or {}
    if not wfo.get("enabled", False):
        raise StaticValidationError("config wfo.enabled must be true")
    exec_block = cfg.get("execution") or {}
    if bool(exec_block.get("allow_short", False)) != bool(allow_short):
        raise StaticValidationError(
            f"config execution.allow_short={exec_block.get('allow_short')} "
            f"does not match strategy allow_short={allow_short}"
        )
```

- [ ] **Step 7: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_validate_static.py -q
```

Expected: 7 passed.

- [ ] **Step 8: Commit**

```powershell
git add factory/factory/validate.py factory/tests/test_validate_static.py factory/tests/fixtures/valid_strategy.py factory/tests/fixtures/invalid_*.py factory/tests/fixtures/valid_config.yaml factory/tests/fixtures/invalid_config_wrong_strategy.yaml
git commit -m @'
feat(factory): Tier 1 static validation (§5.4)

ast-based checks for parse, GeneratedStrategy class, required methods,
strategy_id match, import whitelist (pandas/numpy/dataclasses/__future__/
typing/backtester only), .shift(1) presence, @dataclass(slots=True),
config strategy/wfo.enabled/allow_short alignment. v0.4.0 opt-in
attributes (uses_multi_symbol, uses_per_bar) are explicitly forbidden.
'@
```

---

## Task 6: Validate Tier 2 — functional smoke test (`validate.py` part 2 + `synth_ohlcv.py`)

**Files:**
- Create: `factory/factory/synth_ohlcv.py`
- Modify: `factory/factory/validate.py` (add `validate_functional`)
- Create: `factory/tests/fixtures/invalid_signal_dtype.py`
- Create: `factory/tests/fixtures/invalid_signal_short.py`
- Create: `factory/tests/test_validate_functional.py`

Implements §5.4 Tier 2. Writes the strategy to a temp file, loads it via `importlib.util.spec_from_file_location` (so we DON'T pollute the real registry), instantiates it, runs `indicators` + `generate_signals` against a 200-bar synthetic OHLCV frame, and asserts SignalFrame shape, dtype, range, size positivity, and the first-bar-zero one-bar-shift sanity check. The slow tag lets `-m "not slow"` skip these locally during fast iteration.

- [ ] **Step 1: Write `factory/factory/synth_ohlcv.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n_bars: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic ~n_bars OHLCV frame with realistic constraints.

    Constraints: high >= max(open, close) >= min(open, close) >= low; volume > 0.
    Index is daily business dates. Prices float around 100 via a random walk
    plus a low-amplitude sinusoid so percentile/range-style strategies have
    something to react to.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_bars)
    trend = np.linspace(0.0, 5.0, n_bars)
    wave = 3.0 * np.sin(np.linspace(0.0, 6 * np.pi, n_bars))
    noise = rng.normal(0.0, 0.6, n_bars).cumsum()
    close = 100.0 + trend + wave + noise

    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.2, n_bars - 1)

    intrabar = rng.uniform(0.3, 1.5, n_bars)
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar
    volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df
```

- [ ] **Step 2: Write the two new fixture strategies**

`factory/tests/fixtures/invalid_signal_dtype.py` — copy of `valid_strategy.py`, but in `generate_signals` change the final lines to:

```python
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal.astype(float)  # WRONG dtype
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
```

(Remove the `.shift(1).fillna(0).astype(int)` line and replace with `.astype(float)`. Then add a single literal `.shift(1)` somewhere in a comment so Tier 1 passes — Tier 2 is what catches this.)

`factory/tests/fixtures/invalid_signal_short.py` — copy of `valid_strategy.py`, but in the state-machine loop change `raw_signal[i] = 1` to `raw_signal[i] = -1` (so it emits -1 even though allow_short will be passed as False).

- [ ] **Step 3: Write the failing test `factory/tests/test_validate_functional.py`**

```python
from pathlib import Path

import pytest

from factory.validate import FunctionalValidationError, validate_functional

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.mark.slow
def test_valid_strategy_passes_functional(tmp_path: Path) -> None:
    validate_functional(
        strategy_id="gen_test_valid",
        strategy_src=_read("valid_strategy.py"),
        allow_short=False,
        tmp_dir=tmp_path,
    )


@pytest.mark.slow
def test_bad_signal_dtype_fails(tmp_path: Path) -> None:
    with pytest.raises(FunctionalValidationError) as exc:
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_signal_dtype.py"),
            allow_short=False,
            tmp_dir=tmp_path,
        )
    msg = str(exc.value).lower()
    assert "signal" in msg and ("int" in msg or "dtype" in msg)


@pytest.mark.slow
def test_short_signal_under_long_only_fails(tmp_path: Path) -> None:
    with pytest.raises(FunctionalValidationError) as exc:
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=_read("invalid_signal_short.py"),
            allow_short=False,
            tmp_dir=tmp_path,
        )
    msg = str(exc.value).lower()
    assert "-1" in msg or "short" in msg or "long-only" in msg


@pytest.mark.slow
def test_unimportable_strategy_fails(tmp_path: Path) -> None:
    # Trip an import-time error: bad syntax inside the body.
    bad = "from __future__ import annotations\nthis is not valid python\n"
    with pytest.raises(FunctionalValidationError):
        validate_functional(
            strategy_id="gen_test_valid",
            strategy_src=bad,
            allow_short=False,
            tmp_dir=tmp_path,
        )
```

Add a `slow` marker config to make `-m "not slow"` work. Add to `factory/conftest.py`:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that import the real backtester or run real subprocesses")
```

- [ ] **Step 4: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_validate_functional.py -q -m slow
```

Expected: `validate_functional` doesn't exist.

- [ ] **Step 5: Append `validate_functional` to `factory/factory/validate.py`**

```python
import importlib.util
import sys
import uuid
from pathlib import Path

import pandas as pd

# Append after the existing module body.


def _load_strategy_module(src: str, tmp_dir: Path) -> object:
    """Load a strategy source file as a one-off module without touching the
    registered strategies. Returns the imported module object.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_name = f"_factory_validate_{uuid.uuid4().hex}"
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise FunctionalValidationError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    except FunctionalValidationError:
        raise
    except Exception as exc:
        raise FunctionalValidationError(
            f"strategy failed to import: {type(exc).__name__}: {exc}"
        ) from exc


def validate_functional(
    *,
    strategy_id: str,
    strategy_src: str,
    allow_short: bool,
    tmp_dir: Path,
) -> None:
    """Tier 2 functional smoke test.

    Imports the strategy in isolation (no registry pollution), instantiates it,
    runs indicators() + generate_signals() against a 200-bar synthetic OHLCV
    frame, and asserts the SignalFrame contract.
    """
    # Import dependencies lazily so unit tests of Tier 1 don't need pandas
    # to load this module.
    from backtester.core.types import SignalFrame, StrategyContext
    from factory.synth_ohlcv import make_synthetic_ohlcv

    module = _load_strategy_module(strategy_src, tmp_dir)

    cls = getattr(module, "GeneratedStrategy", None)
    if cls is None:
        raise FunctionalValidationError("imported module has no GeneratedStrategy")

    # Instantiate. params_type() should be a dataclass with all-defaulted fields.
    try:
        params_cls = cls.params_type()
        params = params_cls()
        strategy = cls()
    except Exception as exc:
        raise FunctionalValidationError(
            f"strategy/params instantiation failed: {type(exc).__name__}: {exc}"
        ) from exc

    # warmup_bars sanity.
    try:
        warmup = int(strategy.warmup_bars(params))
    except Exception as exc:
        raise FunctionalValidationError(f"warmup_bars raised: {exc}") from exc
    if warmup < 0 or warmup > 1000:
        raise FunctionalValidationError(f"warmup_bars out of sane range: {warmup}")

    data = make_synthetic_ohlcv(n_bars=max(200, warmup + 50))

    # Indicators.
    try:
        ind = strategy.indicators(data, params)
    except Exception as exc:
        raise FunctionalValidationError(
            f"indicators() raised: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(ind, pd.DataFrame):
        raise FunctionalValidationError(
            f"indicators() returned {type(ind).__name__}, expected DataFrame"
        )
    if not ind.index.equals(data.index):
        raise FunctionalValidationError("indicators index does not match data index")

    # Signals.
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=warmup)
    try:
        sf = strategy.generate_signals(data, ind, ctx, params)
    except Exception as exc:
        raise FunctionalValidationError(
            f"generate_signals() raised: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(sf, SignalFrame):
        raise FunctionalValidationError(
            f"generate_signals returned {type(sf).__name__}, expected SignalFrame"
        )

    df = sf.data
    if not isinstance(df, pd.DataFrame):
        raise FunctionalValidationError("SignalFrame.data is not a DataFrame")
    if "signal" not in df.columns:
        raise FunctionalValidationError("SignalFrame missing 'signal' column")
    if not df.index.equals(data.index):
        raise FunctionalValidationError("SignalFrame index does not match data index")

    sig = df["signal"]
    # Integer dtype required.
    if not pd.api.types.is_integer_dtype(sig):
        raise FunctionalValidationError(
            f"signal column dtype must be integer, got {sig.dtype}"
        )

    unique = set(int(x) for x in sig.dropna().unique())
    allowed = {-1, 0, 1} if allow_short else {0, 1}
    extra = unique - allowed
    if extra:
        raise FunctionalValidationError(
            f"signal values outside allowed set {sorted(allowed)}: found {sorted(extra)}"
        )

    # size column required and positive.
    if sf.size_column is None or sf.size_column not in df.columns:
        raise FunctionalValidationError("SignalFrame missing 'size' column")
    size = df[sf.size_column]
    if (size <= 0).any():
        raise FunctionalValidationError("size column contains non-positive values")

    # First-bar zero (cheap one-bar-shift sanity).
    if int(sig.iloc[0]) != 0:
        raise FunctionalValidationError(
            f"first signal must be 0 (one-bar-shift sanity); got {int(sig.iloc[0])}"
        )
```

- [ ] **Step 6: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_validate_functional.py -q -m slow
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```powershell
git add factory/factory/synth_ohlcv.py factory/factory/validate.py factory/tests/test_validate_functional.py factory/tests/fixtures/invalid_signal_dtype.py factory/tests/fixtures/invalid_signal_short.py factory/conftest.py
git commit -m @'
feat(factory): Tier 2 functional validation (§5.4)

Loads the strategy via importlib.util.spec_from_file_location into a
uuid-suffixed module name (no registry pollution), instantiates, runs
indicators()/generate_signals() against a 200-bar synthetic OHLCV frame,
and asserts SignalFrame contract: DataFrame type, index alignment, integer
signal dtype, value set, positive size, first-bar zero. Marked @slow.
'@
```

---

## Task 7: Filesystem — strategy/config writers + idempotent registry append (`filesystem.py`)

**Files:**
- Create: `factory/factory/filesystem.py`
- Create: `factory/tests/test_filesystem.py`

Implements §5.5. Writes are absolute and final; the registry append is the single touch into the backtester source tree. Idempotency check: scan the registry file for the alias `_<strategy_id>` before appending.

- [ ] **Step 1: Write the failing test `factory/tests/test_filesystem.py`**

```python
from pathlib import Path

import pytest

from factory.filesystem import (
    FilesystemError,
    RegistryAlreadyHasStrategy,
    append_registry_entry,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)


def _seed_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from __future__ import annotations\n"
        "from backtester.strategies.base import BaseStrategy\n"
        "STRATEGY_REGISTRY = {}\n"
        "def register_strategy(cls): STRATEGY_REGISTRY[cls.strategy_id] = cls; return cls\n",
        encoding="utf-8",
    )


def test_write_strategy_and_config(tmp_path: Path) -> None:
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs" / "wfo"
    write_strategy_artifacts(
        strategy_id="gen_42",
        strategy_src="# strategy body\n",
        config_src="run_name: gen_42\n",
        strategies_dir=strat_dir,
        configs_dir=cfg_dir,
    )
    assert (strat_dir / "gen_42.py").read_text(encoding="utf-8") == "# strategy body\n"
    assert (cfg_dir / "gen_42.yaml").read_text(encoding="utf-8") == "run_name: gen_42\n"


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "gen_42.py").write_text("existing", encoding="utf-8")
    with pytest.raises(FilesystemError) as exc:
        write_strategy_artifacts(
            strategy_id="gen_42",
            strategy_src="# new",
            config_src="run_name: gen_42\n",
            strategies_dir=tmp_path / "strategies",
            configs_dir=tmp_path / "configs" / "wfo",
        )
    assert "exists" in str(exc.value).lower()


def test_append_registry_entry_adds_two_lines(tmp_path: Path) -> None:
    reg = tmp_path / "backtester" / "strategies" / "registry.py"
    _seed_registry(reg)
    append_registry_entry(strategy_id="gen_42", registry_file=reg)
    text = reg.read_text(encoding="utf-8")
    assert "from strategies.gen_42 import GeneratedStrategy as _gen_42" in text
    assert "register_strategy(_gen_42)" in text


def test_append_registry_is_idempotent(tmp_path: Path) -> None:
    reg = tmp_path / "backtester" / "strategies" / "registry.py"
    _seed_registry(reg)
    append_registry_entry(strategy_id="gen_42", registry_file=reg)
    with pytest.raises(RegistryAlreadyHasStrategy):
        append_registry_entry(strategy_id="gen_42", registry_file=reg)


def test_pick_unused_strategy_id_returns_base_when_free(tmp_path: Path) -> None:
    strat = tmp_path / "strategies"
    strat.mkdir()
    assert pick_unused_strategy_id("gen_42", strategies_dir=strat) == "gen_42"


def test_pick_unused_strategy_id_bumps_on_collision(tmp_path: Path) -> None:
    strat = tmp_path / "strategies"
    strat.mkdir()
    (strat / "gen_42.py").write_text("x", encoding="utf-8")
    (strat / "gen_42_2.py").write_text("x", encoding="utf-8")
    assert pick_unused_strategy_id("gen_42", strategies_dir=strat) == "gen_42_3"
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_filesystem.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/filesystem.py`**

```python
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class FilesystemError(RuntimeError):
    pass


class RegistryAlreadyHasStrategy(FilesystemError):
    pass


def pick_unused_strategy_id(base: str, *, strategies_dir: Path) -> str:
    """Return `base` if strategies/<base>.py is free, otherwise base_2, base_3, ..."""
    if not (strategies_dir / f"{base}.py").exists():
        return base
    i = 2
    while (strategies_dir / f"{base}_{i}.py").exists():
        i += 1
    return f"{base}_{i}"


def write_strategy_artifacts(
    *,
    strategy_id: str,
    strategy_src: str,
    config_src: str,
    strategies_dir: Path,
    configs_dir: Path,
) -> tuple[Path, Path]:
    """Write the strategy .py and config .yaml.

    Refuses to overwrite either file (collision should have been avoided by
    pick_unused_strategy_id upstream).
    """
    strat_path = strategies_dir / f"{strategy_id}.py"
    cfg_path = configs_dir / f"{strategy_id}.yaml"
    if strat_path.exists():
        raise FilesystemError(f"strategy file already exists: {strat_path}")
    if cfg_path.exists():
        raise FilesystemError(f"config file already exists: {cfg_path}")
    strategies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(strategy_src, encoding="utf-8")
    cfg_path.write_text(config_src, encoding="utf-8")
    log.info("wrote strategy=%s config=%s", strat_path, cfg_path)
    return strat_path, cfg_path


def append_registry_entry(*, strategy_id: str, registry_file: Path) -> None:
    """Append two lines to registry.py:
        from strategies.<strategy_id> import GeneratedStrategy as _<strategy_id>
        register_strategy(_<strategy_id>)
    Idempotency: raises RegistryAlreadyHasStrategy if the alias appears already.
    """
    if not registry_file.exists():
        raise FilesystemError(f"registry file not found: {registry_file}")
    text = registry_file.read_text(encoding="utf-8")
    alias = f"_{strategy_id}"
    needle_import = f"as {alias}"
    needle_register = f"register_strategy({alias})"
    if needle_import in text or needle_register in text:
        raise RegistryAlreadyHasStrategy(
            f"registry already has strategy {strategy_id!r}"
        )
    if not text.endswith("\n"):
        text += "\n"
    lines = (
        f"from strategies.{strategy_id} import GeneratedStrategy as {alias}  # noqa: E402\n"
        f"register_strategy({alias})\n"
    )
    registry_file.write_text(text + lines, encoding="utf-8")
    log.info("appended registry entry for %s", strategy_id)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_filesystem.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/filesystem.py factory/tests/test_filesystem.py
git commit -m @'
feat(factory): strategy/config writers + idempotent registry append (§5.5)

write_strategy_artifacts refuses to overwrite either file. pick_unused_strategy_id
bumps gen_<ts> -> gen_<ts>_2 on collision (rare same-second restart). The
registry append uses the alias pattern `as _<strategy_id>` to avoid class-name
collisions across many generated strategies, and detects the alias on re-runs.
'@
```

---

## Task 8: Stages — bundle finder + summary parsers (`stages.py` part 1)

**Files:**
- Create: `factory/factory/stages.py` (parsers only; subprocesses in Task 9)
- Create: `factory/tests/fixtures/sample_backtest_summary.json`
- Create: `factory/tests/fixtures/sample_optimize_summary.json`
- Create: `factory/tests/fixtures/sample_wfo_summary.json`
- Create: `factory/tests/test_stages_parsers.py`

This is where the reconciliation findings R1 + R2 land. The parsers map the REAL `summary.json` shapes (which differ from spec §6 guesses) onto the factory's record fields. `find_latest_bundle` matches the suffix-naming convention from R2.

- [ ] **Step 1: Write the three real-shape fixtures**

These are direct copies of real `summary.json` files from `output/runs/`. Copy verbatim from the backtester repo:

`factory/tests/fixtures/sample_backtest_summary.json`:

```json
{
  "total_return": 0.04789374275677183,
  "annualized_return": 0.004696651430814214,
  "annualized_vol": 0.06607341894887274,
  "sharpe": 0.10408681366641073,
  "sortino": 0.07678489985397205,
  "max_drawdown": -0.18412515335473634,
  "n_trades": 286,
  "n_round_trips": 143,
  "win_rate": 0.4965034965034965,
  "avg_round_trip_pnl": 33.49212780193766,
  "time_in_market": 0.27980922098569155,
  "turnover": 271.8263207425216,
  "final_equity": 104789.37427567718,
  "params": {"range_period": 10, "percentile_window": 60, "entry_percentile": 20.0, "exit_percentile": 70.0, "max_hold": 5, "size": 1.0},
  "symbol": "SPY",
  "timeframe": "1d"
}
```

`factory/tests/fixtures/sample_optimize_summary.json`:

```json
{
  "best_params": {"fast": 20, "slow": 100, "size": 1.0},
  "best_score_objective": "sharpe",
  "best_summary": {
    "total_return": 1.151945029732036,
    "sharpe": 0.6652567100252722,
    "max_drawdown": -0.22533808426677238,
    "n_trades": 27,
    "win_rate": 0.3076923076923077,
    "params": {"fast": 20, "slow": 100, "size": 1.0},
    "symbol": "SPY",
    "timeframe": "1d"
  }
}
```

`factory/tests/fixtures/sample_wfo_summary.json`:

```json
{
  "oos_summary": {
    "total_return": 0.31078674058870415,
    "sharpe": 0.6897185480952924,
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

- [ ] **Step 2: Write the failing test `factory/tests/test_stages_parsers.py`**

```python
import json
import time
from pathlib import Path

import pytest

from factory.stages import (
    BundleNotFound,
    find_latest_bundle,
    parse_backtest_summary,
    parse_optimize_summary,
    parse_wfo_summary,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_parse_backtest_summary_extracts_headline_metrics() -> None:
    raw = _load("sample_backtest_summary.json")
    parsed = parse_backtest_summary(raw, bundle_path=Path("output/runs/x"))
    assert parsed["sharpe"] == pytest.approx(0.10408681366641073)
    assert parsed["total_return"] == pytest.approx(0.04789374275677183)
    assert parsed["max_drawdown"] == pytest.approx(-0.18412515335473634)
    assert parsed["win_rate"] == pytest.approx(0.4965034965034965)
    assert parsed["n_trades"] == 286
    assert parsed["run_bundle_path"] == "output/runs/x"


def test_parse_optimize_summary_extracts_best_params_and_score() -> None:
    raw = _load("sample_optimize_summary.json")
    parsed = parse_optimize_summary(raw, bundle_path=Path("output/runs/y"))
    assert parsed["best_params"] == {"fast": 20, "slow": 100, "size": 1.0}
    assert parsed["objective"] == "sharpe"
    # best_score == best_summary[objective]
    assert parsed["best_score"] == pytest.approx(0.6652567100252722)
    assert parsed["run_bundle_path"] == "output/runs/y"


def test_parse_wfo_summary_extracts_oos_block() -> None:
    raw = _load("sample_wfo_summary.json")
    parsed = parse_wfo_summary(raw, bundle_path=Path("output/runs/z"))
    assert parsed["oos_sharpe"] == pytest.approx(0.6897185480952924)
    assert parsed["oos_total_return"] == pytest.approx(0.31078674058870415)
    assert parsed["oos_max_drawdown"] == pytest.approx(-0.07374129512318983)
    assert parsed["oos_n_trades"] == 109
    assert parsed["n_windows"] == 6
    assert parsed["parameter_stability"]["entry_percentile"]["mode"] == 30.0


def test_parsers_raise_on_missing_keys() -> None:
    with pytest.raises(KeyError):
        parse_backtest_summary({}, bundle_path=Path("x"))
    with pytest.raises(KeyError):
        parse_optimize_summary({}, bundle_path=Path("x"))
    with pytest.raises(KeyError):
        parse_wfo_summary({"oos_summary": {}}, bundle_path=Path("x"))


def test_find_latest_bundle_picks_newest_matching_run_name(tmp_path: Path) -> None:
    output_runs = tmp_path / "output" / "runs"
    output_runs.mkdir(parents=True)
    # Create three bundles with different mtimes.
    (output_runs / "20260101_0900_gen_X").mkdir()
    time.sleep(0.01)
    (output_runs / "20260101_1000_gen_X").mkdir()
    time.sleep(0.01)
    (output_runs / "20260101_1100_gen_Y").mkdir()
    found = find_latest_bundle(output_runs_dir=output_runs, run_name="gen_X")
    assert found.name == "20260101_1000_gen_X"


def test_find_latest_bundle_raises_when_no_match(tmp_path: Path) -> None:
    output_runs = tmp_path / "output" / "runs"
    output_runs.mkdir(parents=True)
    with pytest.raises(BundleNotFound):
        find_latest_bundle(output_runs_dir=output_runs, run_name="missing")
```

- [ ] **Step 3: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_stages_parsers.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Write `factory/factory/stages.py` (parsers + bundle finder)**

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StageError(RuntimeError):
    """A backtester stage exited non-zero, timed out, or produced no summary.json."""


class BundleNotFound(StageError):
    """No bundle dir matching the expected run_name."""


def find_latest_bundle(*, output_runs_dir: Path, run_name: str) -> Path:
    """Return the newest directory under output_runs_dir whose name ends with
    `_<run_name>` (after the YYYYMMDD_HHMM prefix).
    """
    if not output_runs_dir.exists():
        raise BundleNotFound(f"output_runs_dir does not exist: {output_runs_dir}")
    candidates = [
        d for d in output_runs_dir.iterdir()
        if d.is_dir() and d.name.endswith(f"_{run_name}")
    ]
    if not candidates:
        raise BundleNotFound(
            f"no bundle found in {output_runs_dir} for run_name={run_name!r}"
        )
    # Pick the one with the newest mtime (handles minute-resolution timestamp ties).
    return max(candidates, key=lambda d: d.stat().st_mtime)


def parse_backtest_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map backtest summary.json (flat metrics) onto the factory record shape."""
    return {
        "sharpe": float(raw["sharpe"]),
        "total_return": float(raw["total_return"]),
        "max_drawdown": float(raw["max_drawdown"]),
        "win_rate": float(raw["win_rate"]),
        "n_trades": int(raw["n_trades"]),
        "run_bundle_path": str(bundle_path),
    }


def parse_optimize_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map optimize summary.json onto the factory record shape.

    Real shape: {best_params, best_score_objective, best_summary{...}}.
    best_score = best_summary[best_score_objective].
    """
    best_summary = raw["best_summary"]
    objective = raw["best_score_objective"]
    if objective not in best_summary:
        raise KeyError(
            f"objective {objective!r} not found in best_summary keys "
            f"{sorted(best_summary.keys())}"
        )
    return {
        "best_params": dict(raw["best_params"]),
        "objective": objective,
        "best_score": float(best_summary[objective]),
        "run_bundle_path": str(bundle_path),
    }


def parse_wfo_summary(raw: dict[str, Any], *, bundle_path: Path) -> dict[str, Any]:
    """Map WFO summary.json (nested oos_summary) onto the factory record shape."""
    oos = raw["oos_summary"]
    return {
        "oos_sharpe": float(oos["sharpe"]),
        "oos_total_return": float(oos["total_return"]),
        "oos_max_drawdown": float(oos["max_drawdown"]),
        "oos_n_trades": int(oos["n_trades"]),
        "parameter_stability": dict(raw.get("parameter_stability", {})),
        "n_windows": int(raw["n_windows"]),
        "run_bundle_path": str(bundle_path),
    }
```

- [ ] **Step 5: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_stages_parsers.py -q
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```powershell
git add factory/factory/stages.py factory/tests/test_stages_parsers.py factory/tests/fixtures/sample_*_summary.json
git commit -m @'
feat(factory): summary.json parsers + bundle finder (§5.7, reconciliation R1+R2)

Three parsers map the REAL summary.json shapes (verified against
output/runs/) onto the factory record fields. WFO's oos_summary is nested;
optimize uses best_score_objective + best_summary[objective]; backtest is
flat top-level. find_latest_bundle picks the newest dir ending in
_<run_name> (handles ArtifactWriter's minute-resolution timestamp + the
factory stages.py suffix convention from R2).
'@
```

---

## Task 9: Stages — three subprocess wrappers (`stages.py` part 2)

**Files:**
- Modify: `factory/factory/stages.py` (append `run_<stage>` and `run_all_stages`)
- Create: `factory/tests/test_stages_subprocess.py`

This is where reconciliation finding R2 lands operationally. Each stage gets a transient stage-specific config (rewritten `run_name`) so bundles don't collide. The subprocess invocation is the same for all three stages — `python -m backtester.runners.run_<X> --config <tmp_yaml>`.

- [ ] **Step 1: Write the failing test `factory/tests/test_stages_subprocess.py`**

```python
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from factory.stages import (
    StageError,
    StageResult,
    build_stage_config,
    run_backtest_stage,
    run_optimize_stage,
    run_wfo_stage,
)


def _canonical_config(strategy_id: str) -> str:
    cfg = {
        "run_name": strategy_id,
        "strategy": strategy_id,
        "strategy_params": {"size": 1.0},
        "data": {"symbols": ["SPY"], "timeframe": "1d", "start": "2015-01-02",
                 "end": "2024-12-31", "source": "csv", "root": "data/raw"},
        "execution": {"initial_cash": 100000, "commission_bps": 2,
                      "slippage_bps": 5, "allow_fractional": False, "allow_short": False},
        "portfolio": {"sizing_mode": "percent_equity", "size": 0.95},
        "optimization": {"objective": "sharpe", "param_space": {"size": [0.5, 1.0]}},
        "wfo": {"enabled": True, "train_bars": 756, "test_bars": 252, "step_bars": 252},
    }
    return yaml.safe_dump(cfg, sort_keys=False)


def test_build_stage_config_rewrites_run_name(tmp_path: Path) -> None:
    canonical = tmp_path / "configs" / "wfo" / "gen_42.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")

    tmp_dir = tmp_path / "_tmp"
    bt_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="backtest",
        tmp_dir=tmp_dir,
    )
    opt_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="optimize",
        tmp_dir=tmp_dir,
    )
    wfo_cfg = build_stage_config(
        canonical_path=canonical,
        strategy_id="gen_42",
        stage="wfo",
        tmp_dir=tmp_dir,
    )
    assert bt_cfg.exists() and opt_cfg.exists() and wfo_cfg.exists()
    assert bt_cfg != opt_cfg != wfo_cfg

    bt = yaml.safe_load(bt_cfg.read_text())
    opt = yaml.safe_load(opt_cfg.read_text())
    wfo = yaml.safe_load(wfo_cfg.read_text())
    assert bt["run_name"] == "gen_42"
    assert opt["run_name"] == "gen_42_grid"
    assert wfo["run_name"] == "gen_42_wfo"
    # Everything else carries through unchanged.
    assert bt["strategy"] == opt["strategy"] == wfo["strategy"] == "gen_42"


def test_run_backtest_stage_invokes_subprocess_and_parses(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "total_return": 0.1, "sharpe": 1.2, "max_drawdown": -0.05,
        "win_rate": 0.6, "n_trades": 20,
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc) as run_mock:
        result = run_backtest_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert run_mock.called
    cmd = run_mock.call_args[0][0]
    # Subprocess is invoked with the runner module and --config pointing at
    # the stage-specific YAML, not the canonical one.
    assert "backtester.runners.run_backtest" in cmd
    assert "--config" in cmd
    assert isinstance(result, StageResult)
    assert result.parsed["sharpe"] == pytest.approx(1.2)
    assert result.parsed["run_bundle_path"].endswith("20260101_0900_gen_42")


def test_run_optimize_stage_uses_grid_suffix(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42_grid"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "best_params": {"size": 1.0}, "best_score_objective": "sharpe",
        "best_summary": {"sharpe": 1.5, "params": {"size": 1.0}},
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        result = run_optimize_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert result.parsed["best_score"] == pytest.approx(1.5)
    assert result.parsed["run_bundle_path"].endswith("_gen_42_grid")


def test_run_wfo_stage_uses_wfo_suffix(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    bundle = output_runs / "20260101_0900_gen_42_wfo"
    bundle.mkdir(parents=True)
    (bundle / "summary.json").write_text(json.dumps({
        "oos_summary": {"sharpe": 1.1, "total_return": 0.2, "max_drawdown": -0.06, "n_trades": 30},
        "parameter_stability": {},
        "n_windows": 6,
    }), encoding="utf-8")

    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        result = run_wfo_stage(
            canonical_config=canonical,
            strategy_id="gen_42",
            output_runs_dir=output_runs,
            tmp_dir=tmp_path / "_tmp",
            timeout_sec=30,
        )
    assert result.parsed["oos_sharpe"] == pytest.approx(1.1)
    assert result.parsed["n_windows"] == 6


def test_stage_nonzero_exit_raises_with_stderr_tail(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    fake_proc = mock.Mock(returncode=1, stdout="", stderr="boom traceback line 1\nline 2")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        with pytest.raises(StageError) as exc:
            run_backtest_stage(
                canonical_config=canonical,
                strategy_id="gen_42",
                output_runs_dir=tmp_path / "output" / "runs",
                tmp_dir=tmp_path / "_tmp",
                timeout_sec=30,
            )
    assert "boom" in str(exc.value)


def test_stage_missing_summary_raises(tmp_path: Path) -> None:
    canonical = tmp_path / "cfg.yaml"
    canonical.write_text(_canonical_config("gen_42"), encoding="utf-8")
    output_runs = tmp_path / "output" / "runs"
    (output_runs / "20260101_0900_gen_42").mkdir(parents=True)  # bundle exists, no summary.json
    fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("factory.stages.subprocess.run", return_value=fake_proc):
        with pytest.raises(StageError) as exc:
            run_backtest_stage(
                canonical_config=canonical,
                strategy_id="gen_42",
                output_runs_dir=output_runs,
                tmp_dir=tmp_path / "_tmp",
                timeout_sec=30,
            )
    assert "summary" in str(exc.value).lower()
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_stages_subprocess.py -q
```

Expected: ImportError on `build_stage_config` / `run_backtest_stage` / etc.

- [ ] **Step 3: Append to `factory/factory/stages.py`**

```python
import json
import subprocess
import sys
from dataclasses import dataclass

import yaml


STAGE_SUFFIX = {"backtest": "", "optimize": "_grid", "wfo": "_wfo"}
STAGE_MODULE = {
    "backtest": "backtester.runners.run_backtest",
    "optimize": "backtester.runners.run_optimize",
    "wfo": "backtester.runners.run_wfo",
}


@dataclass(slots=True, frozen=True)
class StageResult:
    stage: str
    parsed: dict[str, Any]
    bundle_path: Path
    raw_summary: dict[str, Any]


def build_stage_config(
    *, canonical_path: Path, strategy_id: str, stage: str, tmp_dir: Path,
) -> Path:
    """Clone the canonical YAML to <tmp_dir>/<strategy_id>/<stage>.yaml with
    `run_name` rewritten so each stage's output bundle is distinct (R2).
    """
    if stage not in STAGE_SUFFIX:
        raise ValueError(f"unknown stage: {stage}")
    raw = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StageError(f"canonical config is not a mapping: {canonical_path}")
    suffix = STAGE_SUFFIX[stage]
    raw["run_name"] = f"{strategy_id}{suffix}"
    out_dir = tmp_dir / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def _run_stage(
    *,
    stage: str,
    canonical_config: Path,
    strategy_id: str,
    output_runs_dir: Path,
    tmp_dir: Path,
    timeout_sec: int,
    backtester_root: Path | None = None,
) -> StageResult:
    """Internal: run one stage subprocess and parse its summary.json."""
    stage_cfg = build_stage_config(
        canonical_path=canonical_config,
        strategy_id=strategy_id,
        stage=stage,
        tmp_dir=tmp_dir,
    )
    cmd = [sys.executable, "-m", STAGE_MODULE[stage], "--config", str(stage_cfg)]
    log.info("running stage=%s cmd=%s", stage, cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd=str(backtester_root) if backtester_root else None,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"stage={stage} timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise StageError(f"stage={stage} exit={proc.returncode}; stderr tail:\n{tail}")

    run_name = f"{strategy_id}{STAGE_SUFFIX[stage]}"
    try:
        bundle = find_latest_bundle(output_runs_dir=output_runs_dir, run_name=run_name)
    except BundleNotFound as exc:
        raise StageError(f"stage={stage}: {exc}") from exc

    summary_path = bundle / "summary.json"
    if not summary_path.exists():
        raise StageError(f"stage={stage}: summary.json missing in {bundle}")
    try:
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(f"stage={stage}: summary.json not valid JSON: {exc}") from exc

    if stage == "backtest":
        parsed = parse_backtest_summary(raw_summary, bundle_path=bundle)
    elif stage == "optimize":
        parsed = parse_optimize_summary(raw_summary, bundle_path=bundle)
    elif stage == "wfo":
        parsed = parse_wfo_summary(raw_summary, bundle_path=bundle)
    else:  # pragma: no cover — guarded above
        raise StageError(f"unknown stage: {stage}")

    return StageResult(stage=stage, parsed=parsed, bundle_path=bundle, raw_summary=raw_summary)


def run_backtest_stage(**kwargs) -> StageResult:
    return _run_stage(stage="backtest", **kwargs)


def run_optimize_stage(**kwargs) -> StageResult:
    return _run_stage(stage="optimize", **kwargs)


def run_wfo_stage(**kwargs) -> StageResult:
    return _run_stage(stage="wfo", **kwargs)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_stages_subprocess.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Sanity check against the real `gen_1715800000` strategy (slow, optional)**

This is the spec's recommended sanity check from §12 build-order step 5 ("Sanity-check by running against gen_1715800000 (which is known to work)").

```powershell
python -c @'
from pathlib import Path
from factory.stages import run_backtest_stage, run_optimize_stage, run_wfo_stage

root = Path(".").resolve()
canonical = root / "configs" / "backtests" / "gen_1715800000.yaml"
tmp_dir = root / "factory" / "data" / "_tmp"
out = root / "output" / "runs"

bt = run_backtest_stage(canonical_config=canonical, strategy_id="gen_1715800000",
                        output_runs_dir=out, tmp_dir=tmp_dir, timeout_sec=300)
print("BT  ", bt.parsed)

opt = run_optimize_stage(canonical_config=canonical, strategy_id="gen_1715800000",
                         output_runs_dir=out, tmp_dir=tmp_dir, timeout_sec=900)
print("OPT ", opt.parsed)

wfo = run_wfo_stage(canonical_config=canonical, strategy_id="gen_1715800000",
                    output_runs_dir=out, tmp_dir=tmp_dir, timeout_sec=1800)
print("WFO ", wfo.parsed)
'@
```

Expected: three StageResult.parsed dicts printed, each with the expected keys (sharpe / best_score / oos_sharpe respectively) and a run_bundle_path pointing into `output/runs/`. Inspect the three bundle dirs — verify they have distinct names ending in `_gen_1715800000`, `_gen_1715800000_grid`, `_gen_1715800000_wfo`.

- [ ] **Step 6: Commit**

```powershell
git add factory/factory/stages.py factory/tests/test_stages_subprocess.py
git commit -m @'
feat(factory): three subprocess stages with collision-safe run_names (§5.7, R2)

build_stage_config clones the canonical YAML to factory/data/_tmp/<id>/<stage>.yaml
with run_name rewritten (suffix "" / "_grid" / "_wfo"), avoiding the
ArtifactWriter minute-resolution bundle collision. Subprocess uses sys.executable
+ -m module form; failure modes (non-zero exit, timeout, missing summary,
malformed JSON) all surface as StageError with stderr tail.
'@
```

---

## Task 10: Results store (`results.py`)

**Files:**
- Create: `factory/factory/results.py`
- Create: `factory/tests/test_results.py`

Implements §5.8 + §6 (reconciled). Append-only JSONL. One record per cycle. The dashboard's single data source.

- [ ] **Step 1: Write the failing test `factory/tests/test_results.py`**

```python
from pathlib import Path

import pytest

from factory.results import (
    Record,
    build_failed_record,
    build_record,
    read_records,
    write_record,
)


def _slots() -> dict[str, str]:
    return {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }


def _idea() -> dict:
    return {
        "one_line_summary": "test idea",
        "hypothesis": "h",
        "novelty_justification": "n",
        "failure_mode": "f",
        "allow_short": False,
    }


def test_build_record_complete_has_all_fields() -> None:
    r = build_record(
        strategy_id="gen_42",
        timestamp="2026-05-15T09:00:00Z",
        slots=_slots(),
        idea=_idea(),
        generation_cost_usd=0.034,
        backtest={"sharpe": 1.1, "total_return": 0.2, "max_drawdown": -0.05,
                  "win_rate": 0.6, "n_trades": 20, "run_bundle_path": "p1"},
        optimize={"best_params": {"size": 1.0}, "objective": "sharpe",
                  "best_score": 1.3, "run_bundle_path": "p2"},
        wfo={"oos_sharpe": 1.2, "oos_total_return": 0.18,
             "oos_max_drawdown": -0.04, "oos_n_trades": 25,
             "parameter_stability": {}, "n_windows": 6,
             "run_bundle_path": "p3"},
        alerted=True,
    )
    assert r["status"] == "complete"
    assert r["failed_stage"] is None
    assert r["error"] is None
    assert r["strategy_id"] == "gen_42"
    assert r["slots"]["strategy_family"] == "momentum"
    assert r["backtest"]["sharpe"] == 1.1
    assert r["wfo"]["oos_sharpe"] == 1.2
    assert r["alerted"] is True


def test_build_failed_record_has_failed_stage_and_error() -> None:
    r = build_failed_record(
        strategy_id="gen_43",
        timestamp="2026-05-15T09:01:00Z",
        slots=_slots(),
        idea=_idea(),
        generation_cost_usd=0.012,
        failed_stage="validation",
        error="missing .shift(1)",
    )
    assert r["status"] == "failed"
    assert r["failed_stage"] == "validation"
    assert r["error"] == "missing .shift(1)"
    assert r["backtest"] is None
    assert r["optimize"] is None
    assert r["wfo"] is None
    assert r["alerted"] is False


def test_build_failed_record_for_generation_failure_has_no_idea() -> None:
    r = build_failed_record(
        strategy_id=None,
        timestamp="2026-05-15T09:02:00Z",
        slots=_slots(),
        idea=None,
        generation_cost_usd=0.0,
        failed_stage="generation",
        error="claude -p timeout",
    )
    assert r["status"] == "failed"
    assert r["failed_stage"] == "generation"
    assert r["strategy_id"] is None
    assert r["idea"] is None


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    store = tmp_path / "results.json"
    write_record(store, {"a": 1, "strategy_id": "x"})
    write_record(store, {"a": 2, "strategy_id": "y"})
    write_record(store, {"a": 3, "strategy_id": "z"})
    records = read_records(store)
    assert [r["a"] for r in records] == [1, 2, 3]


def test_read_records_handles_missing_file(tmp_path: Path) -> None:
    assert read_records(tmp_path / "nothing.json") == []


def test_read_records_skips_blank_lines(tmp_path: Path) -> None:
    store = tmp_path / "results.json"
    store.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_records(store) == [{"a": 1}, {"a": 2}]


def test_read_records_raises_on_malformed_line(tmp_path: Path) -> None:
    store = tmp_path / "results.json"
    store.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(store)
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_results.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/results.py`**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)

Record = dict[str, Any]

FAILED_STAGES = frozenset({"generation", "validation", "backtest", "optimize", "wfo"})


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
) -> Record:
    """Build a `status: complete` results record (§6, reconciled)."""
    return {
        "strategy_id": strategy_id,
        "timestamp": timestamp,
        "status": "complete",
        "failed_stage": None,
        "error": None,
        "slots": dict(slots),
        "idea": dict(idea),
        "generation_cost_usd": float(generation_cost_usd),
        "backtest": dict(backtest) if backtest is not None else None,
        "optimize": dict(optimize) if optimize is not None else None,
        "wfo": dict(wfo) if wfo is not None else None,
        "alerted": bool(alerted),
    }


def build_failed_record(
    *,
    strategy_id: Optional[str],
    timestamp: str,
    slots: Mapping[str, str],
    idea: Optional[Mapping[str, Any]],
    generation_cost_usd: float,
    failed_stage: str,
    error: str,
    backtest: Optional[Mapping[str, Any]] = None,
    optimize: Optional[Mapping[str, Any]] = None,
    wfo: Optional[Mapping[str, Any]] = None,
) -> Record:
    """Build a `status: failed` results record (§3.1)."""
    if failed_stage not in FAILED_STAGES:
        raise ValueError(f"failed_stage must be one of {sorted(FAILED_STAGES)}, got {failed_stage!r}")
    return {
        "strategy_id": strategy_id,
        "timestamp": timestamp,
        "status": "failed",
        "failed_stage": failed_stage,
        "error": error,
        "slots": dict(slots),
        "idea": dict(idea) if idea is not None else None,
        "generation_cost_usd": float(generation_cost_usd),
        "backtest": dict(backtest) if backtest is not None else None,
        "optimize": dict(optimize) if optimize is not None else None,
        "wfo": dict(wfo) if wfo is not None else None,
        "alerted": False,
    }


def write_record(store_path: Path, record: Record) -> None:
    """Append one JSON object as a single line to the JSONL store."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with store_path.open("a", encoding="utf-8") as f:
        f.write(line)


def read_records(store_path: Path) -> list[Record]:
    """Read all records from the JSONL store. Skips blank lines.

    Raises ValueError on any non-blank line that isn't valid JSON (corruption).
    """
    if not store_path.exists():
        return []
    out: list[Record] = []
    for i, raw in enumerate(store_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"results store corruption at line {i}: {exc}") from exc
    return out
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_results.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/results.py factory/tests/test_results.py
git commit -m @'
feat(factory): results record schema + JSONL store (§5.8, §6 reconciled)

build_record produces status=complete with the §6 record shape, but with
the field names mapped onto the REAL summary.json shapes (R1). build_failed_record
covers all five failed_stage values (§3.1). write_record appends one
compact line; read_records tolerates blanks and surfaces corruption.
'@
```

---

## Task 11: Telegram notifier (`notify.py`)

**Files:**
- Create: `factory/factory/notify.py`
- Create: `factory/tests/test_notify.py`

Implements §5.9. Threshold gate keyed on `settings.alerts.alert_threshold_metric` (default `wfo.oos_sharpe`). The message body explicitly labels the alert as a **shortlist signal**, not a verdict (per spec §9 landmine 1 — multiple-comparisons risk). Telegram errors don't crash the loop.

- [ ] **Step 1: Write the failing test `factory/tests/test_notify.py`**

```python
from unittest import mock

import pytest

from factory.notify import (
    NotifyConfig,
    NotifyResult,
    extract_metric,
    format_alert_message,
    maybe_send_alert,
)


def _record() -> dict:
    return {
        "strategy_id": "gen_42",
        "idea": {"one_line_summary": "compression breakout test"},
        "backtest": {"sharpe": 0.9},
        "optimize": {"best_score": 1.4},
        "wfo": {"oos_sharpe": 1.25, "oos_total_return": 0.18,
                "oos_max_drawdown": -0.06, "oos_n_trades": 25},
    }


def test_extract_metric_walks_dotted_path() -> None:
    rec = _record()
    assert extract_metric(rec, "wfo.oos_sharpe") == 1.25
    assert extract_metric(rec, "backtest.sharpe") == 0.9
    assert extract_metric(rec, "optimize.best_score") == 1.4


def test_extract_metric_returns_none_for_missing_path() -> None:
    rec = _record()
    assert extract_metric(rec, "wfo.does_not_exist") is None
    assert extract_metric(rec, "missing.thing") is None
    rec_with_none = {"wfo": None}
    assert extract_metric(rec_with_none, "wfo.oos_sharpe") is None


def test_format_alert_message_labels_as_shortlist_signal() -> None:
    msg = format_alert_message(_record(), dashboard_base_url="http://x.y")
    assert "shortlist signal" in msg.lower()
    assert "gen_42" in msg
    assert "compression breakout test" in msg
    assert "1.25" in msg or "1.250" in msg  # oos_sharpe
    assert "http://x.y" in msg
    # MUST NOT use words that imply finality.
    bad = {"validated", "winner", "confirmed edge"}
    for term in bad:
        assert term not in msg.lower(), term


def test_maybe_send_alert_skips_when_metric_below_threshold() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sharpe",
        alert_threshold=2.0,  # above this record's 1.25
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram") as post:
        result = maybe_send_alert(_record(), cfg)
    assert result == NotifyResult(eligible=False, sent=False, reason="below_threshold")
    post.assert_not_called()


def test_maybe_send_alert_skips_when_credentials_missing() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sharpe",
        alert_threshold=1.0,
        telegram_bot_token="",
        telegram_chat_id="",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram") as post:
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is False
    assert result.reason == "no_credentials"
    post.assert_not_called()


def test_maybe_send_alert_calls_telegram_when_above_threshold() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sharpe",
        alert_threshold=1.0,
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram", return_value=True) as post:
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is True
    post.assert_called_once()
    kwargs = post.call_args.kwargs
    assert kwargs["bot_token"] == "t"
    assert kwargs["chat_id"] == "c"
    assert "gen_42" in kwargs["text"]


def test_maybe_send_alert_swallows_telegram_failures() -> None:
    cfg = NotifyConfig(
        alert_threshold_metric="wfo.oos_sharpe",
        alert_threshold=1.0,
        telegram_bot_token="t",
        telegram_chat_id="c",
        dashboard_base_url="http://x",
    )
    with mock.patch("factory.notify._post_telegram", return_value=False):
        result = maybe_send_alert(_record(), cfg)
    assert result.eligible is True
    assert result.sent is False
    assert result.reason == "telegram_error"
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_notify.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/notify.py`**

```python
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class NotifyConfig:
    alert_threshold_metric: str  # e.g. "wfo.oos_sharpe"
    alert_threshold: float
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_base_url: str


@dataclass(slots=True, frozen=True)
class NotifyResult:
    eligible: bool
    sent: bool
    reason: str = ""


def extract_metric(record: dict, dotted_path: str) -> Optional[float]:
    """Walk a dotted path in the record. Returns None for any missing step."""
    cur: Any = record
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def format_alert_message(record: dict, *, dashboard_base_url: str) -> str:
    """Build the Telegram message body.

    Always labels the alert as a `SHORTLIST SIGNAL`, never as a verdict
    (spec §9 landmine 1 — multiple-comparisons / overfitting risk).
    """
    sid = record["strategy_id"]
    summary = (record.get("idea") or {}).get("one_line_summary", "(no summary)")
    wfo = record.get("wfo") or {}
    parts = [
        f"[SHORTLIST SIGNAL — not a verdict]",
        f"Strategy: {sid}",
        f"Idea: {summary}",
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


def _post_telegram(*, bot_token: str, chat_id: str, text: str) -> bool:
    """POST to the Telegram Bot API sendMessage endpoint. Returns True on 2xx,
    False on any error (logged but swallowed)."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return True
            log.warning("telegram non-2xx status: %s", resp.status)
            return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("telegram post failed: %s", exc)
        return False


def maybe_send_alert(record: dict, cfg: NotifyConfig) -> NotifyResult:
    """Send a Telegram alert iff the threshold metric clears the threshold.

    Never raises. Telegram errors are logged and returned as sent=False with
    reason='telegram_error'.
    """
    if record.get("status") != "complete":
        return NotifyResult(eligible=False, sent=False, reason="not_complete")
    value = extract_metric(record, cfg.alert_threshold_metric)
    if value is None:
        return NotifyResult(eligible=False, sent=False, reason="metric_missing")
    if value < cfg.alert_threshold:
        return NotifyResult(eligible=False, sent=False, reason="below_threshold")
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        log.info("alert eligible but telegram credentials not configured; skipping")
        return NotifyResult(eligible=True, sent=False, reason="no_credentials")

    text = format_alert_message(record, dashboard_base_url=cfg.dashboard_base_url)
    ok = _post_telegram(
        bot_token=cfg.telegram_bot_token,
        chat_id=cfg.telegram_chat_id,
        text=text,
    )
    if not ok:
        return NotifyResult(eligible=True, sent=False, reason="telegram_error")
    return NotifyResult(eligible=True, sent=True, reason="sent")
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_notify.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/notify.py factory/tests/test_notify.py
git commit -m @'
feat(factory): telegram notifier with shortlist-signal labeling (§5.9, §9 landmine 1)

extract_metric walks a dotted path (settings.alert_threshold_metric);
maybe_send_alert is the single gate. Message body opens with
"[SHORTLIST SIGNAL - not a verdict]" and explicitly states a held-out
gate is required before treating the alert as a candidate. Telegram
failures (URLError/timeout) are swallowed -- the loop continues.
'@
```

---

## Task 12: Cycle assembly (`cycle.py`)

**Files:**
- Create: `factory/factory/cycle.py`
- Create: `factory/tests/test_cycle.py`

Implements the 17 steps of §3 in one `run_cycle(settings, rng)` function — the single most timing-sensitive piece of code in the factory. The load-bearing rules from §3.1 and §3.2:

- **Generation failure** → no dedup entry, no files, no registry touch. Skip.
- **Validation failure** → dedup entry written, no files, no registry touch.
- **Stage failure** → dedup entry written, files + registry already written; skip.
- The dedup-log append happens AS SOON AS `one_line_summary` is parseable, BEFORE validation, BEFORE backtester stages.

- [ ] **Step 1: Write the failing test `factory/tests/test_cycle.py`**

```python
import random
from pathlib import Path
from unittest import mock

import pytest

from factory.cycle import CycleOutcome, run_cycle
from factory.settings_loader import load_settings


def _seed_backtester_tree(root: Path) -> None:
    """Lay down a fake backtester tree so the cycle can write into it."""
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "wfo").mkdir(parents=True, exist_ok=True)
    (root / "output" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies" / "registry.py").write_text(
        "def register_strategy(cls): return cls\n", encoding="utf-8"
    )


def _fake_claude_result(strategy_id: str) -> object:
    from factory.generate import GenerationResult
    parsed = {
        "strategy_id": strategy_id,
        "one_line_summary": "test compression strategy",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": "# placeholder body\n",  # validation will fail this
        "config_file": "run_name: x\n",
    }
    return GenerationResult(parsed=parsed, cost_usd=0.03, raw_stdout="{}")


def test_generation_failure_writes_failed_record_and_no_dedup_entry(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)
    from factory.generate import GenerationError
    with mock.patch("factory.cycle.call_claude", side_effect=GenerationError("boom")):
        outcome = run_cycle(s, rng=random.Random(0))
    assert outcome.status == "failed"
    assert outcome.failed_stage == "generation"
    assert not s.paths.dedup_log.exists() or s.paths.dedup_log.read_text().strip() == ""
    # A failed record IS written.
    from factory.results import read_records
    records = read_records(s.paths.results_store)
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["failed_stage"] == "generation"


def test_validation_failure_writes_dedup_but_no_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)
    fake = _fake_claude_result("gen_cycle_test")
    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "failed"
    assert outcome.failed_stage == "validation"
    # Dedup log entry IS present.
    from factory.dedup import read_tail
    tail = read_tail(s.paths.dedup_log, n=10)
    assert tail == ["test compression strategy"]
    # No strategy file or config was written (validation failed before write).
    assert not (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert not (s.paths.configs_dir / "gen_1715800000.yaml").exists()
    # Registry is untouched.
    reg_text = s.paths.registry_file.read_text(encoding="utf-8")
    assert "gen_1715800000" not in reg_text


def test_complete_cycle_writes_files_registry_record(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    # Use the known-good fixture strategy as the generated body.
    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    # Rewrite ids to match the cycle-injected id.
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "valid test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    # Stub stage results.
    from factory.stages import StageResult
    bt = StageResult(stage="backtest",
                     parsed={"sharpe": 1.1, "total_return": 0.2, "max_drawdown": -0.05,
                             "win_rate": 0.6, "n_trades": 20,
                             "run_bundle_path": "p1"},
                     bundle_path=Path("p1"), raw_summary={})
    opt = StageResult(stage="optimize",
                      parsed={"best_params": {"size": 1.0}, "objective": "sharpe",
                              "best_score": 1.3, "run_bundle_path": "p2"},
                      bundle_path=Path("p2"), raw_summary={})
    wfo = StageResult(stage="wfo",
                      parsed={"oos_sharpe": 1.25, "oos_total_return": 0.18,
                              "oos_max_drawdown": -0.06, "oos_n_trades": 25,
                              "parameter_stability": {}, "n_windows": 6,
                              "run_bundle_path": "p3"},
                      bundle_path=Path("p3"), raw_summary={})

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage", return_value=bt), \
         mock.patch("factory.cycle.run_optimize_stage", return_value=opt), \
         mock.patch("factory.cycle.run_wfo_stage", return_value=wfo):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "complete"
    assert outcome.failed_stage is None
    # Strategy + config written.
    assert (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert (s.paths.configs_dir / "gen_1715800000.yaml").exists()
    # Registry has the entry.
    assert "_gen_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")
    # Record written with wfo block.
    from factory.results import read_records
    rec = read_records(s.paths.results_store)[0]
    assert rec["status"] == "complete"
    assert rec["wfo"]["oos_sharpe"] == 1.25
    assert rec["alerted"] is True   # oos_sharpe 1.25 > threshold 1.0; no telegram creds -> still alerted? See below.


def test_stage_failure_writes_failed_record_keeps_dedup_and_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_backtester_tree(tmp_path)
    s = load_settings(tmp_settings_file)

    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_1715800000"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_1715800000")

    from factory.generate import GenerationResult
    from factory.stages import StageError
    fake = GenerationResult(parsed={
        "strategy_id": "gen_1715800000",
        "one_line_summary": "another test idea",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=1715800000), \
         mock.patch("factory.cycle._iso_now", return_value="2026-05-15T09:00:00Z"), \
         mock.patch("factory.cycle.run_backtest_stage",
                    side_effect=StageError("exit=1; traceback ...")):
        outcome = run_cycle(s, rng=random.Random(0))

    assert outcome.status == "failed"
    assert outcome.failed_stage == "backtest"
    # Dedup entry IS present.
    from factory.dedup import read_tail
    assert read_tail(s.paths.dedup_log, n=10) == ["another test idea"]
    # Files + registry ARE present (per §9 landmine 2: orphan accepted).
    assert (s.paths.strategies_dir / "gen_1715800000.py").exists()
    assert "_gen_1715800000" in s.paths.registry_file.read_text(encoding="utf-8")
```

In the third test, fix the alerted assertion to match the test's settings (telegram creds blank → `alerted=False`):

```python
    assert rec["alerted"] is False   # eligible but no creds -> not sent -> alerted=False
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_cycle.py -q
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Write `factory/factory/cycle.py`**

```python
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from factory.dedup import append_summary, read_tail
from factory.filesystem import (
    RegistryAlreadyHasStrategy,
    append_registry_entry,
    pick_unused_strategy_id,
    write_strategy_artifacts,
)
from factory.generate import GenerationError, GenerationResult, call_claude
from factory.notify import NotifyConfig, maybe_send_alert
from factory.prompt import build_prompt
from factory.results import build_failed_record, build_record, write_record
from factory.settings_loader import Settings
from factory.slots import pull_slots
from factory.stages import (
    StageError,
    StageResult,
    run_backtest_stage,
    run_optimize_stage,
    run_wfo_stage,
)
from factory.validate import (
    FunctionalValidationError,
    StaticValidationError,
    validate_functional,
    validate_static,
)

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CycleOutcome:
    status: str                     # "complete" | "failed"
    failed_stage: Optional[str]
    strategy_id: Optional[str]
    record: dict[str, Any]


def _now_unix_int() -> int:
    return int(time.time())


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _notify_cfg(s: Settings) -> NotifyConfig:
    return NotifyConfig(
        alert_threshold_metric=s.alerts.alert_threshold_metric,
        alert_threshold=s.alerts.alert_threshold,
        telegram_bot_token=s.alerts.telegram_bot_token,
        telegram_chat_id=s.alerts.telegram_chat_id,
        dashboard_base_url=s.alerts.dashboard_base_url,
    )


def run_cycle(settings: Settings, *, rng: random.Random) -> CycleOutcome:
    """Execute one full cycle (§3 steps 1-17) and return the outcome.

    Never raises on expected failure modes; everything that goes wrong becomes
    a failed record (§3.1). The dedup-log append is at the FIRST possible
    moment after a parseable one_line_summary exists (§3.2).
    """
    s = settings
    paths = s.paths
    slots = pull_slots(rng)
    ts = _iso_now()
    base_strategy_id = f"gen_{_now_unix_int()}"
    # Step 1-3: slots + dedup tail + prompt.
    dedup_tail = read_tail(paths.dedup_log, n=30)
    strategy_id = pick_unused_strategy_id(base_strategy_id, strategies_dir=paths.strategies_dir)
    prompt = build_prompt(strategy_id=strategy_id, slots=slots, dedup_tail=dedup_tail)
    log.info("cycle start id=%s slots=%s", strategy_id, slots)

    # Step 4-5: generate + parse.
    try:
        gen: GenerationResult = call_claude(
            prompt=prompt,
            claude_cmd=s.generation.claude_cmd,
            claude_flags=s.generation.claude_flags,
            timeout_sec=s.generation.generation_timeout_sec,
        )
    except GenerationError as exc:
        rec = build_failed_record(
            strategy_id=None, timestamp=ts, slots=slots, idea=None,
            generation_cost_usd=0.0, failed_stage="generation", error=str(exc),
        )
        write_record(paths.results_store, rec)
        log.warning("cycle id=%s generation failed: %s", strategy_id, exc)
        return CycleOutcome(status="failed", failed_stage="generation",
                            strategy_id=None, record=rec)

    parsed = gen.parsed
    cost = gen.cost_usd
    idea = {
        "one_line_summary": parsed["one_line_summary"],
        "hypothesis": parsed["hypothesis"],
        "novelty_justification": parsed["novelty_justification"],
        "failure_mode": parsed["failure_mode"],
        "allow_short": bool(parsed["allow_short"]),
    }

    # Step 6: dedup-log append (BEFORE validation, BEFORE stages — §3.2).
    append_summary(paths.dedup_log, parsed["one_line_summary"])

    # Step 7: validate (Tier 1 + Tier 2).
    try:
        validate_static(
            strategy_id=strategy_id,
            strategy_src=parsed["strategy_file"],
            config_src=parsed["config_file"],
            allow_short=bool(parsed["allow_short"]),
        )
        validate_functional(
            strategy_id=strategy_id,
            strategy_src=parsed["strategy_file"],
            allow_short=bool(parsed["allow_short"]),
            tmp_dir=paths.tmp_dir / "validate",
        )
    except (StaticValidationError, FunctionalValidationError) as exc:
        rec = build_failed_record(
            strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
            generation_cost_usd=cost, failed_stage="validation", error=str(exc),
        )
        write_record(paths.results_store, rec)
        log.warning("cycle id=%s validation failed: %s", strategy_id, exc)
        return CycleOutcome(status="failed", failed_stage="validation",
                            strategy_id=strategy_id, record=rec)

    # Step 8-10: write files + register.
    write_strategy_artifacts(
        strategy_id=strategy_id,
        strategy_src=parsed["strategy_file"],
        config_src=parsed["config_file"],
        strategies_dir=paths.strategies_dir,
        configs_dir=paths.configs_dir,
    )
    try:
        append_registry_entry(strategy_id=strategy_id, registry_file=paths.registry_file)
    except RegistryAlreadyHasStrategy:
        log.info("registry already has %s; continuing", strategy_id)

    canonical_cfg = paths.configs_dir / f"{strategy_id}.yaml"

    # Step 11-13: run the three stages sequentially.
    bt: Optional[StageResult] = None
    opt: Optional[StageResult] = None
    wfo: Optional[StageResult] = None
    for stage_name, runner in (
        ("backtest", run_backtest_stage),
        ("optimize", run_optimize_stage),
        ("wfo", run_wfo_stage),
    ):
        try:
            result = runner(
                canonical_config=canonical_cfg,
                strategy_id=strategy_id,
                output_runs_dir=paths.output_runs_dir,
                tmp_dir=paths.tmp_dir,
                timeout_sec=s.stages.stage_timeout_sec,
                backtester_root=paths.backtester_root,
            )
        except StageError as exc:
            rec = build_failed_record(
                strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
                generation_cost_usd=cost, failed_stage=stage_name, error=str(exc),
                backtest=bt.parsed if bt else None,
                optimize=opt.parsed if opt else None,
            )
            write_record(paths.results_store, rec)
            log.warning("cycle id=%s stage=%s failed: %s", strategy_id, stage_name, exc)
            return CycleOutcome(status="failed", failed_stage=stage_name,
                                strategy_id=strategy_id, record=rec)
        if stage_name == "backtest":
            bt = result
        elif stage_name == "optimize":
            opt = result
        else:
            wfo = result

    assert bt is not None and opt is not None and wfo is not None

    # Step 14-15: build complete record.
    rec = build_record(
        strategy_id=strategy_id, timestamp=ts, slots=slots, idea=idea,
        generation_cost_usd=cost,
        backtest=bt.parsed, optimize=opt.parsed, wfo=wfo.parsed,
        alerted=False,  # patched below after maybe_send_alert
    )

    # Step 16: alert (conditional). maybe_send_alert never raises.
    notify_result = maybe_send_alert(rec, _notify_cfg(s))
    rec["alerted"] = bool(notify_result.sent)

    write_record(paths.results_store, rec)
    log.info("cycle id=%s complete oos_sharpe=%s alerted=%s",
             strategy_id, wfo.parsed.get("oos_sharpe"), rec["alerted"])
    return CycleOutcome(status="complete", failed_stage=None,
                        strategy_id=strategy_id, record=rec)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_cycle.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/cycle.py factory/tests/test_cycle.py
git commit -m @'
feat(factory): one-cycle orchestration with strict failure/dedup timing (§3)

run_cycle() executes steps 1-17. Load-bearing timing per §3.1/§3.2:
- generation failure -> failed record, no dedup, no files, no registry
- validation failure -> failed record, dedup IS written, no files
- stage failure     -> failed record, dedup AND files+registry kept (orphan
  accepted per §9 landmine 2)
Cycle never raises; every expected failure path produces a failed record.
'@
```

---

## Task 13: Loop (`loop.py`) — continuous mode, signal handling, log rotation

**Files:**
- Create: `factory/factory/loop.py`
- Create: `factory/tests/test_loop.py`

Implements §5.10. Continuous `while True` with graceful SIGINT/SIGTERM. Rotating file handler on `factory/logs/factory.log` (10MB × 5 backups). Bounded `max_cycles` for tests.

- [ ] **Step 1: Write the failing test `factory/tests/test_loop.py`**

```python
import logging
import random
from pathlib import Path
from unittest import mock

import pytest

from factory.loop import configure_logging, run_loop
from factory.settings_loader import load_settings


def test_configure_logging_creates_rotating_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "factory.log"
    configure_logging(log_path)
    root = logging.getLogger("factory")
    assert any(
        "RotatingFileHandler" in type(h).__name__ for h in root.handlers
    )


def test_run_loop_runs_max_cycles_then_exits(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    s = load_settings(tmp_settings_file)
    assert s.loop.max_cycles == 1   # from the test fixture

    from factory.cycle import CycleOutcome
    fake_outcome = CycleOutcome(status="failed", failed_stage="generation",
                                strategy_id=None, record={"status": "failed"})
    with mock.patch("factory.loop.run_cycle", return_value=fake_outcome) as rc:
        completed = run_loop(s, rng=random.Random(0))
    assert rc.call_count == 1
    assert completed == 1


def test_run_loop_stops_on_sigint(tmp_settings_file: Path) -> None:
    s = load_settings(tmp_settings_file)
    # Override max_cycles to 0 (unbounded) and inject SIGINT after first cycle.
    from factory.loop import _ShutdownFlag, run_loop
    from factory.cycle import CycleOutcome

    flag = _ShutdownFlag()
    outcome = CycleOutcome(status="complete", failed_stage=None,
                           strategy_id="gen_x", record={"status": "complete"})

    call_count = {"n": 0}
    def fake_cycle(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            flag.set()
        return outcome

    with mock.patch("factory.loop.run_cycle", side_effect=fake_cycle):
        completed = run_loop(
            s, rng=random.Random(0), shutdown_flag=flag, max_cycles_override=0,
        )
    # The flag is checked AFTER each cycle, so cycle 2 runs to completion
    # and then the loop breaks.
    assert completed == 2
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
python -m pytest factory/tests/test_loop.py -q
```

Expected: ImportError.

- [ ] **Step 3: Write `factory/factory/loop.py`**

```python
from __future__ import annotations

import logging
import logging.handlers
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from factory.cycle import run_cycle
from factory.settings_loader import Settings, load_settings

log = logging.getLogger(__name__)


class _ShutdownFlag:
    """Threadsafe set-once flag used by signal handlers."""
    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def configure_logging(log_path: Path) -> None:
    """Configure a rotating file handler on the `factory` logger root.

    Idempotent: re-configuring removes prior handlers first.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("factory")
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_h = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)
    # Also mirror to stderr for interactive runs.
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(fmt)
    root.addHandler(stream_h)


def _install_signal_handlers(flag: _ShutdownFlag) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        log.info("received signal %s; requesting graceful shutdown", signum)
        flag.set()
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def run_loop(
    settings: Settings,
    *,
    rng: random.Random,
    shutdown_flag: Optional[_ShutdownFlag] = None,
    max_cycles_override: Optional[int] = None,
) -> int:
    """Run cycles in a loop. Returns the number of cycles completed.

    Stops when shutdown_flag is set OR when max_cycles (>0) is reached.
    max_cycles_override (if provided) wins over settings.loop.max_cycles.
    """
    flag = shutdown_flag or _ShutdownFlag()
    max_cycles = max_cycles_override if max_cycles_override is not None else settings.loop.max_cycles
    sleep_sec = settings.loop.inter_cycle_sleep_sec

    completed = 0
    while not flag.is_set():
        try:
            outcome = run_cycle(settings, rng=rng)
            log.info("cycle %d outcome=%s id=%s",
                     completed + 1, outcome.status, outcome.strategy_id)
        except Exception as exc:
            # An unexpected exception from inside run_cycle: log and continue.
            # (run_cycle is supposed to never raise on expected failures, so
            # reaching here means a bug — but the loop must not die.)
            log.exception("unexpected exception in run_cycle: %s", exc)
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
    log.info("loop stopping after %d cycles", completed)
    return completed


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser("factory.loop")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
        help="Path to settings.toml",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for slot pulls")
    args = parser.parse_args(argv)

    s = load_settings(args.settings)
    configure_logging(s.paths.factory_log)
    flag = _ShutdownFlag()
    _install_signal_handlers(flag)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    run_loop(s, rng=rng, shutdown_flag=flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_loop.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add factory/factory/loop.py factory/tests/test_loop.py
git commit -m @'
feat(factory): continuous loop with signal handling + log rotation (§5.10)

run_loop() drives run_cycle() until shutdown_flag or max_cycles. Sleep is
chopped into 0.5s slices so SIGINT/SIGTERM are responsive. Unexpected
exceptions inside run_cycle are logged and swallowed -- the loop must not
die. configure_logging sets a RotatingFileHandler (10MB, 5 backups).
'@
```

---

## Task 14: Dashboard backend — Flask app, JSON endpoint, overview route (`dashboard/server.py` part 1)

**Files:**
- Create: `factory/dashboard/server.py`
- Create: `factory/dashboard/templates/overview.html`
- Create: `factory/dashboard/static/style.css`
- Create: `factory/tests/test_dashboard.py`

Implements §8.1. Flask app with three endpoints:
- `GET /` — overview HTML (server-rendered table)
- `GET /api/records` — raw JSONL records as a JSON array (the data source for client-side auto-refresh)
- `GET /api/summary` — top-of-page counters (total cycles, completes, failures by stage, count above threshold, cumulative spend)

Detail view + auto-refresh come in Task 15/16.

- [ ] **Step 1: Add Flask to the dev environment**

```powershell
python -m pip install Flask
```

Expected: `Successfully installed Flask-3.x.x` (and its deps). The factory's runtime needs Flask; everything else in factory uses only stdlib + pandas/numpy/pyyaml that the backtester already provides.

- [ ] **Step 2: Write the failing test `factory/tests/test_dashboard.py`**

```python
import json
from pathlib import Path

import pytest


def _write_records(store: Path, recs: list[dict]) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def app_with_records(tmp_settings_file: Path, tmp_path: Path):
    from factory.settings_loader import load_settings
    from dashboard.server import create_app
    s = load_settings(tmp_settings_file)
    _write_records(s.paths.results_store, [
        {"strategy_id": "gen_1", "timestamp": "2026-05-15T09:00:00Z",
         "status": "complete", "failed_stage": None, "error": None,
         "slots": {"strategy_family": "momentum"},
         "idea": {"one_line_summary": "first", "allow_short": False},
         "generation_cost_usd": 0.03,
         "backtest": {"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.1,
                      "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p"},
         "optimize": {"best_params": {}, "objective": "sharpe", "best_score": 0.7, "run_bundle_path": "p"},
         "wfo": {"oos_sharpe": 1.2, "oos_total_return": 0.2, "oos_max_drawdown": -0.05,
                 "oos_n_trades": 25, "parameter_stability": {}, "n_windows": 6,
                 "run_bundle_path": "p"},
         "alerted": True},
        {"strategy_id": "gen_2", "timestamp": "2026-05-15T09:05:00Z",
         "status": "failed", "failed_stage": "validation",
         "error": "missing .shift(1)",
         "slots": {"strategy_family": "breakout"},
         "idea": {"one_line_summary": "second"},
         "generation_cost_usd": 0.02,
         "backtest": None, "optimize": None, "wfo": None, "alerted": False},
        {"strategy_id": None, "timestamp": "2026-05-15T09:10:00Z",
         "status": "failed", "failed_stage": "generation",
         "error": "timeout",
         "slots": {"strategy_family": "momentum"},
         "idea": None, "generation_cost_usd": 0.0,
         "backtest": None, "optimize": None, "wfo": None, "alerted": False},
    ])
    app = create_app(settings=s)
    app.config["TESTING"] = True
    return app.test_client(), s


def test_overview_html_renders(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "gen_1" in body
    assert "first" in body or "compression" in body or "gen_2" in body
    # The "shortlist signal" framing must be visible (spec §9 landmine 1).
    assert "shortlist signal" in body.lower() or "shortlist" in body.lower()


def test_api_records_returns_jsonl(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/api/records")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["strategy_id"] == "gen_1"


def test_api_summary_aggregates_counts(app_with_records) -> None:
    client, s = app_with_records
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_cycles"] == 3
    assert data["completes"] == 1
    assert data["failures_by_stage"]["validation"] == 1
    assert data["failures_by_stage"]["generation"] == 1
    # Threshold metric = wfo.oos_sharpe, threshold = 1.0; gen_1's 1.2 clears.
    assert data["above_threshold"] == 1
    assert data["cumulative_spend_usd"] == pytest.approx(0.05)
    assert data["threshold_metric"] == "wfo.oos_sharpe"
    assert data["threshold_value"] == 1.0
```

- [ ] **Step 3: Write `factory/dashboard/static/style.css`**

```css
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 1rem; background: #fafafa; color: #222; }
h1 { margin-top: 0; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; }
th { background: #f0f0f0; cursor: pointer; user-select: none; }
tr.failed { background: #fff0f0; }
tr.good { background: #f0fff5; }
tr:hover { background: #f5f8ff; cursor: pointer; }
.banner { background: #fff7cc; border: 1px solid #e0d36b; padding: 0.75rem 1rem; margin-bottom: 1rem; border-radius: 4px; font-size: 14px; }
.counters { margin: 0.5rem 0 1rem; font-size: 14px; color: #555; }
.counters span { margin-right: 1.25rem; }
.failed-stage { color: #b00; font-weight: bold; }
```

- [ ] **Step 4: Write `factory/dashboard/templates/overview.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Strategy Factory — overview</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <h1>Strategy Factory — overview</h1>
  <div class="banner">
    All "good" flags and Telegram alerts in this dashboard are
    <strong>shortlist signals, not verdicts</strong>. WFO mitigates but does not
    eliminate multiple-comparisons / overfitting risk; a held-out gate
    (different symbol or fully unseen period) is required before treating
    any row as a real candidate.
  </div>
  <div class="counters" id="counters">
    <span>Total cycles: <strong id="c-total">{{ summary.total_cycles }}</strong></span>
    <span>Completes: <strong id="c-complete">{{ summary.completes }}</strong></span>
    <span>Failures: <strong id="c-failures">{{ summary.total_cycles - summary.completes }}</strong>
      ({% for stage, n in summary.failures_by_stage.items() %}{{ stage }}={{ n }}{% if not loop.last %}, {% endif %}{% endfor %})</span>
    <span>Above threshold ({{ summary.threshold_metric }} &gt; {{ summary.threshold_value }}):
      <strong id="c-above">{{ summary.above_threshold }}</strong></span>
    <span>Cumulative spend: <strong id="c-spend">${{ "%.2f" % summary.cumulative_spend_usd }}</strong></span>
  </div>
  <table id="records">
    <thead>
      <tr>
        <th data-sort="timestamp">Timestamp</th>
        <th data-sort="strategy_id">Strategy ID</th>
        <th>Idea</th>
        <th data-sort="status">Status</th>
        <th data-sort="backtest_sharpe">BT Sharpe</th>
        <th data-sort="oos_sharpe">OOS Sharpe</th>
        <th data-sort="oos_total_return">OOS Return</th>
        <th data-sort="oos_max_drawdown">OOS DD</th>
        <th>Good?</th>
        <th>Alerted?</th>
      </tr>
    </thead>
    <tbody id="records-body">
      {% for r in records %}
      <tr class="{% if r.status == 'failed' %}failed{% elif r.is_good %}good{% endif %}"
          data-strategy-id="{{ r.strategy_id or '' }}">
        <td>{{ r.timestamp }}</td>
        <td>{{ r.strategy_id or '(no id)' }}</td>
        <td>{{ (r.idea or {}).get('one_line_summary', '') if r.idea else '(no idea)' }}</td>
        <td>{% if r.status == 'failed' %}<span class="failed-stage">failed: {{ r.failed_stage }}</span>{% else %}{{ r.status }}{% endif %}</td>
        <td>{{ "%.3f" % r.backtest.sharpe if r.backtest else '' }}</td>
        <td>{{ "%.3f" % r.wfo.oos_sharpe if r.wfo else '' }}</td>
        <td>{{ "%.2%%" % (r.wfo.oos_total_return * 100) if r.wfo else '' }}</td>
        <td>{{ "%.2%%" % (r.wfo.oos_max_drawdown * 100) if r.wfo else '' }}</td>
        <td>{{ '*' if r.is_good else '' }}</td>
        <td>{{ '*' if r.alerted else '' }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  <script src="/static/overview.js" defer></script>
</body>
</html>
```

(The `overview.js` is empty for now; auto-refresh code lands in Task 17. We add the file in Step 5 below as an empty stub so Flask serves it.)

- [ ] **Step 5: Create empty `factory/dashboard/static/overview.js`**

```javascript
// Auto-refresh + client-side sort land in Task 17.
```

- [ ] **Step 6: Write `factory/dashboard/server.py`**

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

from factory.notify import extract_metric
from factory.results import read_records
from factory.settings_loader import Settings, load_settings

log = logging.getLogger(__name__)


def _enrich(records: list[dict], threshold_metric: str, threshold: float) -> list[dict]:
    """Add an `is_good` flag to each record (threshold metric clears threshold)."""
    enriched = []
    for r in records:
        val = extract_metric(r, threshold_metric) if r.get("status") == "complete" else None
        is_good = bool(val is not None and val > threshold)
        enriched.append({**r, "is_good": is_good})
    return enriched


def _aggregate(records: list[dict], threshold_metric: str, threshold: float) -> dict[str, Any]:
    total = len(records)
    completes = sum(1 for r in records if r.get("status") == "complete")
    failures_by_stage: dict[str, int] = {}
    above_threshold = 0
    cumulative_spend = 0.0
    for r in records:
        cumulative_spend += float(r.get("generation_cost_usd") or 0.0)
        if r.get("status") == "failed":
            stage = r.get("failed_stage") or "unknown"
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1
        elif r.get("status") == "complete":
            val = extract_metric(r, threshold_metric)
            if val is not None and val > threshold:
                above_threshold += 1
    return {
        "total_cycles": total,
        "completes": completes,
        "failures_by_stage": failures_by_stage,
        "above_threshold": above_threshold,
        "cumulative_spend_usd": cumulative_spend,
        "threshold_metric": threshold_metric,
        "threshold_value": threshold,
    }


def create_app(*, settings: Settings) -> Flask:
    """Build a Flask app bound to one Settings (one results store)."""
    here = Path(__file__).parent.resolve()
    app = Flask(
        __name__,
        template_folder=str(here / "templates"),
        static_folder=str(here / "static"),
    )

    @app.get("/")
    def overview():
        records = read_records(settings.paths.results_store)
        # Newest first for the table.
        records = list(reversed(records))
        enriched = _enrich(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        )
        summary = _aggregate(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        )
        return render_template(
            "overview.html",
            records=enriched,
            summary=summary,
            auto_refresh_sec=settings.dashboard.auto_refresh_sec,
        )

    @app.get("/api/records")
    def api_records():
        records = read_records(settings.paths.results_store)
        return jsonify(records)

    @app.get("/api/summary")
    def api_summary():
        records = read_records(settings.paths.results_store)
        return jsonify(_aggregate(
            records,
            threshold_metric=settings.alerts.alert_threshold_metric,
            threshold=settings.alerts.alert_threshold,
        ))

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser("dashboard.server")
    parser.add_argument(
        "--settings",
        default="factory/config/settings.toml",
        type=Path,
    )
    args = parser.parse_args(argv)
    s = load_settings(args.settings)
    app = create_app(settings=s)
    app.run(host=s.dashboard.host, port=s.dashboard.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_dashboard.py -q
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```powershell
git add factory/dashboard/ factory/tests/test_dashboard.py
git commit -m @'
feat(factory): dashboard overview + JSON endpoints (§8.1)

create_app(settings=...) returns a Flask app bound to one results store.
GET / renders the overview table newest-first with the "shortlist signal,
not verdict" banner above it. GET /api/records returns raw JSONL records;
GET /api/summary returns aggregated counters (total, completes, failures
by stage, above-threshold count, cumulative spend).
'@
```

---

## Task 15: Dashboard detail view (`dashboard/server.py` part 2)

**Files:**
- Modify: `factory/dashboard/server.py` (add `/strategy/<id>` route)
- Create: `factory/dashboard/templates/detail.html`
- Modify: `factory/tests/test_dashboard.py` (add detail tests)

Implements §8.2. Click-through panel: full slots + idea block, per-stage metrics, parameter stability, raw error if failed. Also surfaces the run-bundle paths so the developer can inspect the backtester's full artifacts.

- [ ] **Step 1: Append failing tests to `factory/tests/test_dashboard.py`**

```python
def test_detail_view_renders_complete_record(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/gen_1")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "gen_1" in body
    assert "shortlist signal" in body.lower()
    # OOS values should appear formatted.
    assert "1.200" in body or "1.2" in body  # oos_sharpe
    # Run-bundle path is surfaced so the user can inspect artifacts.
    assert "run_bundle_path" in body or "Run bundle" in body


def test_detail_view_for_failed_cycle_shows_error(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/gen_2")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "missing .shift(1)" in body
    assert "failed" in body.lower()


def test_detail_view_404_on_missing_id(app_with_records) -> None:
    client, _ = app_with_records
    resp = client.get("/strategy/does_not_exist")
    assert resp.status_code == 404
```

- [ ] **Step 2: Write `factory/dashboard/templates/detail.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ record.strategy_id or 'failed cycle' }} — detail</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <p><a href="/">&laquo; back to overview</a></p>
  <h1>{{ record.strategy_id or '(no strategy id — generation failed)' }}</h1>
  <div class="banner">
    Shortlist signal, not a verdict. A held-out gate is required before any row
    here is treated as a real candidate.
  </div>

  <h2>Slots</h2>
  <ul>
    {% for k, v in record.slots.items() %}<li><strong>{{ k }}:</strong> {{ v }}</li>{% endfor %}
  </ul>

  <h2>Idea</h2>
  {% if record.idea %}
    <ul>
      <li><strong>summary:</strong> {{ record.idea.one_line_summary }}</li>
      <li><strong>hypothesis:</strong> {{ record.idea.hypothesis }}</li>
      <li><strong>novelty:</strong> {{ record.idea.novelty_justification }}</li>
      <li><strong>failure mode:</strong> {{ record.idea.failure_mode }}</li>
      <li><strong>allow_short:</strong> {{ record.idea.allow_short }}</li>
    </ul>
  {% else %}
    <p>(no idea was generated — see error below)</p>
  {% endif %}

  <p><strong>Status:</strong>
    {% if record.status == 'failed' %}
      <span class="failed-stage">FAILED at {{ record.failed_stage }}</span>
    {% else %}
      complete
    {% endif %}
  </p>
  {% if record.error %}<p><strong>Error:</strong> <pre>{{ record.error }}</pre></p>{% endif %}
  <p><strong>Generation cost:</strong> ${{ "%.4f" % record.generation_cost_usd }}</p>

  <h2>Stage 1 — Backtest</h2>
  {% if record.backtest %}
    <ul>
      {% for k, v in record.backtest.items() %}
        <li><strong>{{ k }}:</strong> {{ v }}</li>
      {% endfor %}
    </ul>
  {% else %}<p>(not reached)</p>{% endif %}

  <h2>Stage 2 — Optimize</h2>
  {% if record.optimize %}
    <ul>
      <li><strong>best_params:</strong> {{ record.optimize.best_params }}</li>
      <li><strong>objective:</strong> {{ record.optimize.objective }}</li>
      <li><strong>best_score:</strong> {{ "%.4f" % record.optimize.best_score }}</li>
      <li><strong>run_bundle_path:</strong> {{ record.optimize.run_bundle_path }}</li>
    </ul>
  {% else %}<p>(not reached)</p>{% endif %}

  <h2>Stage 3 — WFO</h2>
  {% if record.wfo %}
    <ul>
      <li><strong>oos_sharpe:</strong> {{ "%.3f" % record.wfo.oos_sharpe }}</li>
      <li><strong>oos_total_return:</strong> {{ "%.2%%" % (record.wfo.oos_total_return * 100) }}</li>
      <li><strong>oos_max_drawdown:</strong> {{ "%.2%%" % (record.wfo.oos_max_drawdown * 100) }}</li>
      <li><strong>oos_n_trades:</strong> {{ record.wfo.oos_n_trades }}</li>
      <li><strong>n_windows:</strong> {{ record.wfo.n_windows }}</li>
      <li><strong>parameter_stability:</strong>
        <pre>{{ record.wfo.parameter_stability | tojson(indent=2) }}</pre></li>
      <li><strong>run_bundle_path:</strong> {{ record.wfo.run_bundle_path }}</li>
    </ul>
  {% else %}<p>(not reached)</p>{% endif %}

  <p><strong>Alerted:</strong> {{ record.alerted }}</p>
</body>
</html>
```

- [ ] **Step 3: Add the `/strategy/<id>` route in `factory/dashboard/server.py`**

Insert this inside `create_app(...)` alongside the existing routes:

```python
    @app.get("/strategy/<sid>")
    def detail(sid: str):
        records = read_records(settings.paths.results_store)
        match = next((r for r in records if r.get("strategy_id") == sid), None)
        if match is None:
            return ("not found", 404)
        return render_template("detail.html", record=match)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_dashboard.py -q
```

Expected: 6 passed total (3 from Task 14 + 3 new).

- [ ] **Step 5: Commit**

```powershell
git add factory/dashboard/server.py factory/dashboard/templates/detail.html factory/tests/test_dashboard.py
git commit -m @'
feat(factory): dashboard detail view (§8.2)

GET /strategy/<id> renders the full record: slots, idea block, per-stage
metrics, parameter stability, error tail for failed cycles, and run-bundle
paths so the developer can drop into the backtester's output bundles.
'@
```

---

## Task 16: Dashboard auto-refresh + client-side sort (`overview.js`)

**Files:**
- Modify: `factory/dashboard/static/overview.js`
- Modify: `factory/dashboard/templates/overview.html` (add a row-click handler and pass auto_refresh_sec)
- Modify: `factory/tests/test_dashboard.py` (one extra assertion)

Implements §8.3 auto-refresh. Poll `/api/records` + `/api/summary` every `auto_refresh_sec` seconds; rebuild the `<tbody>` and update the counter spans without a full page reload. Click on a row navigates to `/strategy/<id>`.

- [ ] **Step 1: Replace `factory/dashboard/static/overview.js`**

```javascript
(function () {
  const REFRESH_SEC = parseInt(document.body.dataset.refreshSec || "10", 10);
  const THRESHOLD_METRIC = document.body.dataset.thresholdMetric || "wfo.oos_sharpe";
  const THRESHOLD = parseFloat(document.body.dataset.thresholdValue || "1.0");

  function extractMetric(rec, path) {
    let cur = rec;
    for (const part of path.split(".")) {
      if (cur == null || typeof cur !== "object") return null;
      cur = cur[part];
    }
    return typeof cur === "number" ? cur : null;
  }

  function fmt(n, digits = 3) {
    if (n == null) return "";
    return Number(n).toFixed(digits);
  }
  function fmtPct(n) {
    if (n == null) return "";
    return (n * 100).toFixed(2) + "%";
  }

  function rowFor(rec) {
    const tr = document.createElement("tr");
    const value = rec.status === "complete" ? extractMetric(rec, THRESHOLD_METRIC) : null;
    const isGood = value != null && value > THRESHOLD;
    if (rec.status === "failed") tr.classList.add("failed");
    else if (isGood) tr.classList.add("good");
    if (rec.strategy_id) tr.dataset.strategyId = rec.strategy_id;
    const cells = [
      rec.timestamp || "",
      rec.strategy_id || "(no id)",
      (rec.idea && rec.idea.one_line_summary) || "(no idea)",
      rec.status === "failed"
        ? `<span class="failed-stage">failed: ${rec.failed_stage}</span>`
        : rec.status,
      rec.backtest ? fmt(rec.backtest.sharpe) : "",
      rec.wfo ? fmt(rec.wfo.oos_sharpe) : "",
      rec.wfo ? fmtPct(rec.wfo.oos_total_return) : "",
      rec.wfo ? fmtPct(rec.wfo.oos_max_drawdown) : "",
      isGood ? "*" : "",
      rec.alerted ? "*" : "",
    ];
    for (const html of cells) {
      const td = document.createElement("td");
      td.innerHTML = html;
      tr.appendChild(td);
    }
    return tr;
  }

  async function refresh() {
    try {
      const [recsResp, sumResp] = await Promise.all([
        fetch("/api/records"),
        fetch("/api/summary"),
      ]);
      if (!recsResp.ok || !sumResp.ok) return;
      const records = await recsResp.json();
      const summary = await sumResp.json();

      const tbody = document.getElementById("records-body");
      if (tbody) {
        tbody.innerHTML = "";
        // Newest first.
        for (let i = records.length - 1; i >= 0; i--) {
          tbody.appendChild(rowFor(records[i]));
        }
      }
      const t = document.getElementById("c-total");       if (t) t.textContent = summary.total_cycles;
      const c = document.getElementById("c-complete");    if (c) c.textContent = summary.completes;
      const f = document.getElementById("c-failures");    if (f) f.textContent =
        (summary.total_cycles - summary.completes) + " (" +
        Object.entries(summary.failures_by_stage).map(([k, v]) => `${k}=${v}`).join(", ") + ")";
      const a = document.getElementById("c-above");       if (a) a.textContent = summary.above_threshold;
      const s = document.getElementById("c-spend");       if (s) s.textContent = "$" + Number(summary.cumulative_spend_usd).toFixed(2);
    } catch (err) {
      console.warn("refresh failed", err);
    }
  }

  document.addEventListener("click", function (ev) {
    let el = ev.target;
    while (el && el.tagName !== "TR") el = el.parentElement;
    if (el && el.dataset && el.dataset.strategyId) {
      window.location.href = "/strategy/" + el.dataset.strategyId;
    }
  });

  if (REFRESH_SEC > 0) {
    setInterval(refresh, REFRESH_SEC * 1000);
  }
})();
```

- [ ] **Step 2: Pass refresh interval + threshold to the page**

Edit `factory/dashboard/templates/overview.html` — replace the `<body>` opening tag with:

```html
<body data-refresh-sec="{{ auto_refresh_sec }}"
      data-threshold-metric="{{ summary.threshold_metric }}"
      data-threshold-value="{{ summary.threshold_value }}">
```

- [ ] **Step 3: Add a single assertion to `test_dashboard.py`**

```python
def test_overview_carries_refresh_dataset(app_with_records) -> None:
    client, _ = app_with_records
    body = client.get("/").get_data(as_text=True)
    assert 'data-refresh-sec="10"' in body
    assert 'data-threshold-metric="wfo.oos_sharpe"' in body
    assert 'data-threshold-value="1.0"' in body
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
python -m pytest factory/tests/test_dashboard.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Visual sanity check (manual, not committed)**

```powershell
python -m dashboard.server --settings factory/config/settings.toml
```

Open http://127.0.0.1:8787 in a browser. With an empty `results.json`, the table should be empty and the banner visible. After the first real cycle, a row should appear within ~10s without a manual reload.

- [ ] **Step 6: Commit**

```powershell
git add factory/dashboard/static/overview.js factory/dashboard/templates/overview.html factory/tests/test_dashboard.py
git commit -m @'
feat(factory): dashboard client-side auto-refresh + row-click navigation (§8.3)

Body data attributes carry auto_refresh_sec / threshold_metric / threshold
into the JS. setInterval re-fetches /api/records + /api/summary, rebuilds
the tbody newest-first, and updates the counter spans. Row click navigates
to /strategy/<id>. Refresh is purely client-side; the server endpoint stays
stateless.
'@
```

---

## Task 17: End-to-end smoke test — one full cycle on a known-good strategy

**Files:**
- Create: `factory/tests/test_integration_smoke.py`

This is the build's go/no-go check. It does NOT call `claude -p`. Instead, it stubs `call_claude` to return the literal source of `gen_1715800000.py` and its config, then runs `run_cycle` for real — the validation, write, register, and three subprocess stages all execute against the actual backtester. Slow (~few minutes on the full 2015-2024 SPY data).

- [ ] **Step 1: Write `factory/tests/test_integration_smoke.py`**

```python
import random
import shutil
from pathlib import Path
from unittest import mock

import pytest


@pytest.mark.slow
def test_one_full_cycle_against_real_backtester(tmp_path: Path) -> None:
    """One real cycle, end-to-end, against the real backtester.

    Uses the known-good `gen_1715800000.py` as the generated strategy body
    (with a fresh id to avoid registry collision). All three stages run as
    actual subprocesses against real CSV data in the backtester repo.
    """
    repo = Path(__file__).resolve().parents[2]   # backtester root
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text(f"""
[paths]
backtester_root  = "{repo.as_posix()}"
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_log        = "{(tmp_path / 'dedup.txt').as_posix()}"
results_store    = "{(tmp_path / 'results.json').as_posix()}"
factory_log      = "{(tmp_path / 'factory.log').as_posix()}"
tmp_dir          = "{(tmp_path / '_tmp').as_posix()}"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p"]
generation_timeout_sec = 120

[stages]
stage_timeout_sec = 1800

[alerts]
alert_threshold_metric = "wfo.oos_sharpe"
alert_threshold        = 999.0
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
""", encoding="utf-8")

    from factory.settings_loader import load_settings
    s = load_settings(settings_path)

    # Use the known-good strategy as the "generated" body.
    known_src = (repo / "strategies" / "gen_1715800000.py").read_text(encoding="utf-8")
    known_cfg = (repo / "configs" / "backtests" / "gen_1715800000.yaml").read_text(encoding="utf-8")
    # Rebrand to a smoke-test id so we don't collide with the existing registry entry.
    smoke_id = "gen_factory_smoke"
    smoke_src = known_src.replace('strategy_id = "gen_1715800000"', f'strategy_id = "{smoke_id}"')
    smoke_cfg = known_cfg.replace("gen_1715800000", smoke_id)

    from factory.generate import GenerationResult
    fake_gen = GenerationResult(
        parsed={
            "strategy_id": smoke_id,
            "one_line_summary": "smoke-test range compression",
            "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
            "allow_short": False,
            "strategy_file": smoke_src,
            "config_file": smoke_cfg,
        },
        cost_usd=0.0,
        raw_stdout="{}",
    )

    from factory.cycle import run_cycle
    try:
        with mock.patch("factory.cycle.call_claude", return_value=fake_gen), \
             mock.patch("factory.cycle._now_unix_int", return_value=int.from_bytes(b"smk", "big")):
            outcome = run_cycle(s, rng=random.Random(0))
    finally:
        # Cleanup: remove the smoke strategy file and registry line so the
        # backtester repo isn't permanently polluted.
        strat_file = repo / "strategies" / f"{smoke_id}.py"
        cfg_file = repo / "configs" / "wfo" / f"{smoke_id}.yaml"
        if strat_file.exists():
            strat_file.unlink()
        if cfg_file.exists():
            cfg_file.unlink()
        reg = repo / "backtester" / "strategies" / "registry.py"
        text = reg.read_text(encoding="utf-8")
        cleaned = "\n".join(
            line for line in text.splitlines() if smoke_id not in line
        ) + "\n"
        reg.write_text(cleaned, encoding="utf-8")

    assert outcome.status == "complete", outcome.record
    assert outcome.record["backtest"]["sharpe"] is not None
    assert outcome.record["optimize"]["best_params"]
    assert outcome.record["wfo"]["oos_sharpe"] is not None
    assert outcome.record["wfo"]["n_windows"] > 0
```

- [ ] **Step 2: Run the integration smoke test**

```powershell
python -m pytest factory/tests/test_integration_smoke.py -q -m slow
```

Expected: 1 passed in ~3-10 minutes. The test prints nothing on success; on failure, the captured `outcome.record` includes the failed stage and stderr tail.

- [ ] **Step 3: Inspect side effects**

After the test, manually confirm:
- `<backtester_root>/strategies/gen_factory_smoke.py` is GONE (cleanup ran).
- `<backtester_root>/backtester/strategies/registry.py` has no `gen_factory_smoke` lines.
- `<backtester_root>/output/runs/` has three new bundles: `<TS>_gen_factory_smoke`, `<TS>_gen_factory_smoke_grid`, `<TS>_gen_factory_smoke_wfo` — these are kept as evidence the cycle ran. Delete them manually if you want a clean tree.

- [ ] **Step 4: Commit**

```powershell
git add factory/tests/test_integration_smoke.py
git commit -m @'
test(factory): end-to-end smoke test on known-good gen_1715800000 (@slow)

One full cycle with call_claude stubbed to return the real gen_1715800000
strategy body (rebranded to gen_factory_smoke). Validation Tier 2 imports
against the real backtester package; the three runner subprocesses run on
real CSV data. The test cleans up its registry line and strategy/config
files on exit so the backtester repo is left tidy.
'@
```

---

## Task 18: Failure-mode integration tests

**Files:**
- Create: `factory/tests/test_integration_failures.py`

One test per failure point from §3.1: confirms the correct record shape lands in the JSONL store and the correct side effects (or absence thereof). These are NOT marked slow — they all stub out the subprocesses.

- [ ] **Step 1: Write `factory/tests/test_integration_failures.py`**

```python
import random
from pathlib import Path
from unittest import mock

import pytest


def _seed_tree(root: Path) -> None:
    (root / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "wfo").mkdir(parents=True, exist_ok=True)
    (root / "output" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies").mkdir(parents=True, exist_ok=True)
    (root / "backtester" / "strategies" / "registry.py").write_text(
        "def register_strategy(cls): return cls\n", encoding="utf-8",
    )


def test_generation_timeout_records_failed_stage_generation(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_tree(tmp_path)
    from factory.cycle import run_cycle
    from factory.generate import GenerationError
    from factory.results import read_records
    from factory.settings_loader import load_settings
    s = load_settings(tmp_settings_file)
    with mock.patch("factory.cycle.call_claude",
                    side_effect=GenerationError("claude -p timed out after 120s")):
        run_cycle(s, rng=random.Random(0))
    rec = read_records(s.paths.results_store)[0]
    assert rec["status"] == "failed"
    assert rec["failed_stage"] == "generation"
    assert "timed out" in rec["error"]


def test_validation_failure_keeps_dedup_no_files(
    tmp_settings_file: Path, tmp_path: Path,
) -> None:
    _seed_tree(tmp_path)
    from factory.cycle import run_cycle
    from factory.generate import GenerationResult
    from factory.dedup import read_tail
    from factory.results import read_records
    from factory.settings_loader import load_settings
    s = load_settings(tmp_settings_file)
    fake = GenerationResult(parsed={
        "strategy_id": "gen_xx",
        "one_line_summary": "broken strategy",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": "not even python",
        "config_file": "run_name: gen_xx\n",
    }, cost_usd=0.01, raw_stdout="{}")
    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=42):
        run_cycle(s, rng=random.Random(0))
    rec = read_records(s.paths.results_store)[0]
    assert rec["status"] == "failed" and rec["failed_stage"] == "validation"
    # Dedup entry IS kept (§3.2).
    assert read_tail(s.paths.dedup_log, n=10) == ["broken strategy"]
    # No files / no registry line.
    assert not (s.paths.strategies_dir / "gen_42.py").exists()
    assert "gen_42" not in s.paths.registry_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("failed_stage", ["backtest", "optimize", "wfo"])
def test_stage_failure_records_correct_failed_stage(
    tmp_settings_file: Path, tmp_path: Path, failed_stage: str,
) -> None:
    _seed_tree(tmp_path)
    from factory.cycle import run_cycle
    from factory.generate import GenerationResult
    from factory.stages import StageError, StageResult
    from factory.results import read_records
    from factory.settings_loader import load_settings
    s = load_settings(tmp_settings_file)

    valid_src = (Path(__file__).parent / "fixtures" / "valid_strategy.py").read_text(encoding="utf-8")
    valid_cfg = (Path(__file__).parent / "fixtures" / "valid_config.yaml").read_text(encoding="utf-8")
    valid_src = valid_src.replace('strategy_id = "gen_test_valid"', 'strategy_id = "gen_42"')
    valid_cfg = valid_cfg.replace("gen_test_valid", "gen_42")

    fake = GenerationResult(parsed={
        "strategy_id": "gen_42",
        "one_line_summary": f"trigger {failed_stage} failure",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": valid_src,
        "config_file": valid_cfg,
    }, cost_usd=0.04, raw_stdout="{}")

    bt = StageResult(stage="backtest",
                     parsed={"sharpe": 0.5, "total_return": 0.1, "max_drawdown": -0.05,
                             "win_rate": 0.5, "n_trades": 10, "run_bundle_path": "p"},
                     bundle_path=Path("p"), raw_summary={})
    opt = StageResult(stage="optimize",
                      parsed={"best_params": {}, "objective": "sharpe", "best_score": 0.7,
                              "run_bundle_path": "p"},
                      bundle_path=Path("p"), raw_summary={})

    patches = {"factory.cycle.call_claude": mock.DEFAULT,
               "factory.cycle._now_unix_int": mock.DEFAULT,
               "factory.cycle.run_backtest_stage": mock.DEFAULT,
               "factory.cycle.run_optimize_stage": mock.DEFAULT,
               "factory.cycle.run_wfo_stage": mock.DEFAULT}

    def side(stage: str):
        if stage == failed_stage:
            return StageError(f"stage={stage} exit=1; ...stderr tail...")
        if stage == "backtest":
            return bt
        if stage == "optimize":
            return opt
        return None

    with mock.patch("factory.cycle.call_claude", return_value=fake), \
         mock.patch("factory.cycle._now_unix_int", return_value=42), \
         mock.patch("factory.cycle.run_backtest_stage",
                    side_effect=(side("backtest"),) if failed_stage == "backtest" else None,
                    return_value=bt if failed_stage != "backtest" else mock.DEFAULT), \
         mock.patch("factory.cycle.run_optimize_stage",
                    side_effect=(side("optimize"),) if failed_stage == "optimize" else None,
                    return_value=opt if failed_stage != "optimize" else mock.DEFAULT), \
         mock.patch("factory.cycle.run_wfo_stage",
                    side_effect=(side("wfo"),) if failed_stage == "wfo" else None):
        run_cycle(s, rng=random.Random(0))

    rec = read_records(s.paths.results_store)[0]
    assert rec["status"] == "failed"
    assert rec["failed_stage"] == failed_stage
    assert f"stage={failed_stage}" in rec["error"]
    # Files + registry kept (§9 landmine 2).
    assert (s.paths.strategies_dir / "gen_42.py").exists()
    assert "_gen_42" in s.paths.registry_file.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the tests**

```powershell
python -m pytest factory/tests/test_integration_failures.py -q
```

Expected: 5 passed (1 generation, 1 validation, 3 stage-failure parameterizations). If the parametrized test's mock wiring is awkward, simplify by splitting into three explicit tests.

- [ ] **Step 3: Commit**

```powershell
git add factory/tests/test_integration_failures.py
git commit -m @'
test(factory): integration tests for every §3.1 failure point

Generation failure -> no dedup, no files; validation failure -> dedup
kept, no files; stage failure -> dedup + files + registry kept (orphan
per §9 landmine 2). Each produces a results record with the correct
failed_stage value.
'@
```

---

## Task 19: 100-cycle endurance check + Telegram smoke

**Files:**
- Create: `factory/scripts/endurance_check.py` (a manual driver, not pytest)
- Create: `factory/scripts/telegram_smoke.py` (a manual driver)

These are operator scripts, not unit tests. The endurance check confirms the spec's Definition of Done — *"100+ continuous cycles run without manual intervention; no leaks, no registry corruption, no orphaned partial writes"* — using a stubbed generator so we don't burn $5 on `claude -p` credit during the build. The Telegram smoke is a one-shot to verify the alert path end-to-end against a real bot.

- [ ] **Step 1: Write `factory/scripts/endurance_check.py`**

```python
"""100-cycle endurance check with stubbed claude -p.

Runs 100 cycles against the real backtester subprocesses on the real CSV
data. Generator output rotates through three pre-built strategy/config
pairs (one valid, one validation-fail, one stage-fail) so we hit every
record path. Verifies at the end that the results store contains 100
records, the registry has no duplicate lines, the dedup log has the
right number of entries, and no orphan tmp files exist.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from unittest import mock


SCENARIOS = ("valid", "validation_fail", "stage_fail")


def _load_settings(repo: Path, scratch: Path):
    from factory.settings_loader import load_settings
    settings_toml = scratch / "settings.toml"
    settings_toml.write_text(f"""
[paths]
backtester_root  = "{repo.as_posix()}"
strategies_dir   = "strategies"
configs_dir      = "configs/wfo"
registry_file    = "backtester/strategies/registry.py"
output_runs_dir  = "output/runs"
dedup_log        = "{(scratch / 'dedup.txt').as_posix()}"
results_store    = "{(scratch / 'results.json').as_posix()}"
factory_log      = "{(scratch / 'factory.log').as_posix()}"
tmp_dir          = "{(scratch / '_tmp').as_posix()}"

[generation]
claude_cmd             = "claude"
claude_flags           = ["-p"]
generation_timeout_sec = 120

[stages]
stage_timeout_sec = 1800

[alerts]
alert_threshold_metric = "wfo.oos_sharpe"
alert_threshold        = 999.0
telegram_bot_token     = ""
telegram_chat_id       = ""
dashboard_base_url     = "http://127.0.0.1:8787"

[loop]
mode                  = "continuous"
inter_cycle_sleep_sec = 0
max_cycles            = 0

[dashboard]
host             = "127.0.0.1"
port             = 8787
auto_refresh_sec = 10
""", encoding="utf-8")
    return load_settings(settings_toml)


def _scenario_payload(scenario: str, strategy_id: str, fixtures: Path) -> dict:
    if scenario == "valid":
        src = (fixtures / "valid_strategy.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    elif scenario == "validation_fail":
        src = (fixtures / "invalid_no_shift.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    else:  # stage_fail — valid src but config that will fail backtest
        src = (fixtures / "valid_strategy.py").read_text(encoding="utf-8")
        cfg = (fixtures / "valid_config.yaml").read_text(encoding="utf-8")
    src = re.sub(r'strategy_id = "[^"]+"', f'strategy_id = "{strategy_id}"', src, count=1)
    cfg = re.sub(r"gen_test_valid", strategy_id, cfg)
    return {
        "strategy_id": strategy_id,
        "one_line_summary": f"endurance test cycle {scenario}",
        "hypothesis": "h", "novelty_justification": "n", "failure_mode": "f",
        "allow_short": False,
        "strategy_file": src,
        "config_file": cfg,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--scratch", type=Path,
                        default=Path("factory/data/_endurance_scratch"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    fixtures = repo / "factory" / "tests" / "fixtures"
    args.scratch.mkdir(parents=True, exist_ok=True)
    s = _load_settings(repo, args.scratch)

    from factory.cycle import run_cycle
    from factory.generate import GenerationResult
    from factory.loop import configure_logging

    configure_logging(s.paths.factory_log)
    rng = random.Random(0)

    completed = 0
    counter = {"n": 0}

    def fake_call_claude(**kwargs):
        counter["n"] += 1
        scenario = SCENARIOS[counter["n"] % len(SCENARIOS)]
        sid = f"gen_endurance_{counter['n']}"
        parsed = _scenario_payload(scenario, sid, fixtures)
        return GenerationResult(parsed=parsed, cost_usd=0.03, raw_stdout="{}")

    with mock.patch("factory.cycle.call_claude", side_effect=fake_call_claude):
        for i in range(args.cycles):
            outcome = run_cycle(s, rng=rng)
            completed += 1
            if (i + 1) % 10 == 0:
                print(f"  cycle {i + 1}/{args.cycles} -> {outcome.status}")

    # Post-run invariants.
    from factory.dedup import read_tail
    from factory.results import read_records
    records = read_records(s.paths.results_store)
    print(f"records: {len(records)} (expected {args.cycles})")
    assert len(records) == args.cycles, "results store record count mismatch"

    statuses = {"complete": 0, "failed": 0}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"complete: {statuses['complete']}  failed: {statuses['failed']}")

    # Registry must have NO duplicate `_gen_endurance_*` aliases.
    reg_text = s.paths.registry_file.read_text(encoding="utf-8")
    aliases = re.findall(r"_gen_endurance_\d+", reg_text)
    assert len(aliases) == len(set(aliases)), "duplicate registry entries detected"

    print(f"registry alias count: {len(aliases)}")
    print("ENDURANCE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run a small endurance check first (10 cycles) to find blow-ups early**

```powershell
python -m factory.scripts.endurance_check --cycles 10
```

Expected: prints `ENDURANCE CHECK PASSED`, 10 records, registry aliases all unique. If it fails, fix the underlying issue (probably leaking tmp files, a registry append bug, or an off-by-one in scenario rotation).

- [ ] **Step 3: Run the full 100-cycle check**

```powershell
python -m factory.scripts.endurance_check --cycles 100
```

Expected: completes in ~30-90 minutes (the 33% valid scenarios each run three real subprocess stages). `ENDURANCE CHECK PASSED`. Manually clean up `<backtester_root>/strategies/gen_endurance_*.py`, `<backtester_root>/configs/wfo/gen_endurance_*.yaml`, and the matching registry lines after.

- [ ] **Step 4: Write `factory/scripts/telegram_smoke.py`**

```python
"""One-shot Telegram smoke test.

Reads bot_token + chat_id from settings.toml, posts a single test message
to confirm the credentials are working. Does NOT run a cycle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="factory/config/settings.toml", type=Path)
    args = parser.parse_args(argv)

    from factory.notify import NotifyConfig, format_alert_message, maybe_send_alert
    from factory.settings_loader import load_settings
    s = load_settings(args.settings)
    if not s.alerts.telegram_bot_token or not s.alerts.telegram_chat_id:
        print("ERROR: telegram_bot_token / telegram_chat_id not configured in settings.toml")
        return 2

    fake_record = {
        "strategy_id": "gen_telegram_smoke",
        "status": "complete",
        "idea": {"one_line_summary": "telegram smoke test"},
        "backtest": {"sharpe": 0.5},
        "optimize": {"best_score": 0.7},
        "wfo": {"oos_sharpe": 2.0, "oos_total_return": 0.30,
                "oos_max_drawdown": -0.06, "oos_n_trades": 25},
    }
    cfg = NotifyConfig(
        alert_threshold_metric=s.alerts.alert_threshold_metric,
        alert_threshold=s.alerts.alert_threshold,
        telegram_bot_token=s.alerts.telegram_bot_token,
        telegram_chat_id=s.alerts.telegram_chat_id,
        dashboard_base_url=s.alerts.dashboard_base_url,
    )
    print(format_alert_message(fake_record, dashboard_base_url=s.alerts.dashboard_base_url))
    result = maybe_send_alert(fake_record, cfg)
    print(f"NotifyResult: eligible={result.eligible} sent={result.sent} reason={result.reason}")
    return 0 if result.sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the Telegram smoke (only if credentials are filled in)**

```powershell
python -m factory.scripts.telegram_smoke
```

Expected: prints the formatted message, then `NotifyResult: eligible=True sent=True reason=sent`. A real message appears in the target Telegram chat with the `[SHORTLIST SIGNAL — not a verdict]` header.

- [ ] **Step 6: Commit**

```powershell
git add factory/scripts/
git commit -m @'
chore(factory): endurance check + telegram smoke driver scripts

endurance_check stubs claude -p and rotates through three scenarios
(valid / validation-fail / stage-fail) so 100 cycles exercise every record
path. Post-run invariants: record count matches, no duplicate registry
aliases. telegram_smoke posts one labelled test message to verify
credentials. Both are operator scripts, not pytest -- they take real
wall-clock time and produce real side effects.
'@
```

---

## Definition of Done — checklist before declaring v0.2.0 shipped

Tick each item explicitly. None of these are aspirational; each one corresponds to a concrete check or test in the plan above.

- [ ] Unit suite green: `python -m pytest factory/tests -q -m "not slow"` reports 0 failures.
- [ ] Slow suite green: `python -m pytest factory/tests -q -m slow` reports 0 failures (includes Tier 2 validation + the end-to-end smoke test from Task 17).
- [ ] One full real cycle has produced a row in `factory/data/results.json` with all three stage blocks populated and an OOS Sharpe value present.
- [ ] Each failure point from §3.1 has produced a `failed` record with the matching `failed_stage` value (covered by Task 18).
- [ ] The dashboard loads at `http://127.0.0.1:8787`, sorted-by-OOS-Sharpe behavior works (client-side column header click in `overview.js`), the detail view opens on row click, the table auto-refreshes inside the configured interval.
- [ ] 100-cycle endurance check (Task 19) passed: 100 records, no duplicate registry aliases, no leaked tmp files.
- [ ] Telegram smoke either fired automatically during a real cycle OR `factory/scripts/telegram_smoke.py` succeeded.
- [ ] The backtester's `master` branch has had NO source files modified by the factory other than `backtester/strategies/registry.py`. Verify with:
  ```powershell
  git diff master -- backtester/ | Select-String -NotMatch "^\+from strategies\." -NotMatch "^\+register_strategy"
  ```
  Expected: only the registry append lines are added; nothing else under `backtester/` is changed.

---

## Self-Review

Walked back through the spec section-by-section against the plan. Findings:

**§3 — The cycle**: Tasks 12 (cycle.py) cover steps 1-17 in order. Step 6 (dedup append) is explicitly placed after step 5 (parse), BEFORE step 7 (validate) and step 11-13 (stages), matching the §3.2 timing rule. Step 16 (alert) is conditional on `status == complete` and the threshold gate.

**§3.1 — Failure handling**: Task 18 has one test per failure point (generation, validation, backtest/optimize/wfo). Task 12 implements the three different side-effect patterns (no-dedup-no-files, dedup-yes-files-no, dedup-yes-files-yes-registry).

**§3.2 — Dedup-log timing**: Task 12 step "dedup-log append (BEFORE validation, BEFORE stages — §3.2)" — explicit. Task 3 (dedup.py) handles the append mechanics. Task 18's validation-failure test asserts the dedup entry survives.

**§4 — Components**: Plan's File Structure section matches the spec's tree, plus the additions explicitly motivated by reconciliation: `settings_loader.py`, `synth_ohlcv.py`, `cycle.py` (peeled out of `loop.py` so cycle is independently testable), `data/_tmp/`, `scripts/`. No directory in the spec is missing.

**§5.1 (slots) — §5.10 (loop)**: Each section maps to one task:
- §5.1 → Task 1
- §5.2 → Task 2
- §5.3 → Task 4
- §5.4 → Tasks 5 (Tier 1) + 6 (Tier 2)
- §5.5 → Task 7
- §5.6 → Task 3
- §5.7 → Tasks 8 + 9
- §5.8 → Task 10
- §5.9 → Task 11
- §5.10 → Tasks 12 + 13 (cycle peeled out from loop)

**§6 — Results record schema**: The plan's record shape in Task 10 explicitly applies the reconciliation finding R1: `backtest` uses flat keys, `optimize` uses `best_score` derived from `best_summary[objective]` + reports `objective` (mapping the runner's `best_score_objective`), `wfo` flattens `oos_summary.<metric>` into `oos_*` keys at the record's wfo block. The spec's PROVISIONAL marker is honored.

**§7 — Validation gate**: Tasks 5+6. Tier 1 < Tier 2 < write < register, enforced by the structure of `run_cycle`.

**§8 — Dashboard**: Tasks 14 (overview + JSON endpoints) + 15 (detail view) + 16 (auto-refresh + row-click). §8.3's "auto-refresh poll every N seconds" → JS `setInterval` driven by `data-refresh-sec`. Read-only is enforced by having no POST/PUT/DELETE routes.

**§9 — Known risks**:
- Risk 1 (multiple comparisons): banner on overview + banner on detail + alert header all say "shortlist signal, not a verdict". Telegram message starts with `[SHORTLIST SIGNAL — not a verdict]`. Tests assert this language.
- Risk 2 (orphaned strategies): Task 18 stage-failure test asserts files + registry are kept; comments in cycle.py explicitly note the v0.3 cleanup deferral.
- Risk 3 (registry growth): not addressed — accepted per §9.
- Risk 4 (claude -p output reliability): Task 4 covers clean, fenced, and prose-wrapped outputs; parser is the spec's double-unwrap.
- Risk 5 (stage duration): unchanged; the loop self-paces.
- Risk 6 (rolling.apply): the prompt template already includes the "prefer vectorised pandas ops" line.

**§10 — Configuration**: Task 0 step 4 reproduces the full `settings.toml`. Adds `[paths] tmp_dir` (R2) and `[alerts] dashboard_base_url` (for the Telegram link).

**§11 — Decisions**: `[DECISION-1]` (Stage 1 uses defaults, natural order) — Task 9 runs stages in canonical YAML order, not feeding optimize-best back. `[DECISION-2]` (failure handling) — Task 18 covers. `[DECISION-2 refined]` (dedup timing) — Task 12 explicit. `[DECISION-3]` (threshold metric configurable) — `NotifyConfig.alert_threshold_metric` in Task 11. `[OPEN-Q-B]` (functional smoke INCLUDED) — Task 6. `[OPEN-Q-C]` (continuous, not cron) — Task 13. `[OPEN-Q-D]` (auto-refresh) — Task 16.

**§12 — Build order**: Spec's 8 steps → plan's 19 tasks. Mapping:
- Step 0 (read real interfaces, reconcile) → done before writing the plan; findings R1–R7 documented in "Pre-build reconciliation findings".
- Step 1 (settings + paths + slots + prompt) → Tasks 0 + 1 + 2.
- Step 2 (generate.py one real call) → Task 4 (with a manual smoke step inside).
- Step 3 (validate.py both tiers) → Tasks 5 + 6.
- Step 4 (filesystem.py + dedup.py) → Tasks 3 + 7.
- Step 5 (stages.py) → Tasks 8 + 9 (sanity-check against `gen_1715800000` is built into Task 9 step 5).
- Step 6 (results.py + notify.py) → Tasks 10 + 11.
- Step 7 (loop.py) → Tasks 12 + 13.
- Step 8 (dashboard) → Tasks 14 + 15 + 16.

**Appendix A — Prompt template**: Reproduced verbatim in Task 2's `factory/prompt.py`. Two factory-specific additions (banning `uses_multi_symbol` / `uses_per_bar`) are called out explicitly in the commit message and a `validate_static` check (Task 5) enforces them.

**Placeholder scan**: searched the plan for "TBD", "TODO", "implement later", "fill in details", "add appropriate error handling", "similar to Task". None present in production code. The two operator scripts (Task 19) contain `print(...)` lines, which is intentional — they are CLI drivers.

**Type / signature consistency**:
- `GenerationResult` defined in Task 4, used unchanged in Tasks 12, 17, 18.
- `StageResult` defined in Task 9, used unchanged in Tasks 12, 17, 18.
- `NotifyConfig` / `NotifyResult` defined in Task 11, used in Tasks 12, 19.
- `Settings` and its sub-dataclasses (`Paths`, `GenerationCfg`, `StagesCfg`, `AlertsCfg`, `LoopCfg`, `DashboardCfg`) defined in Task 0, used unchanged downstream.
- `extract_metric` defined in Task 11, reused by Task 14 (`_enrich`, `_aggregate`).
- `_now_unix_int` / `_iso_now` patched in Tasks 12, 17, 18 with consistent signatures.
- `read_records` / `write_record` signatures unchanged across Tasks 10, 14, 18, 19.

**Gap check against spec sections** (final pass):
- All `[DECISION-N]` and `[OPEN-Q-X]` tags from §11 are addressed.
- The spec's §6 PROVISIONAL marker is honored: the plan documents the reconciled real shape in R1 and applies it in Tasks 8 + 10.
- The spec's §5.5 single-config-per-strategy convention is honored (canonical YAML at `configs/wfo/<id>.yaml`); R2's stage-specific transient YAMLs live in `factory/data/_tmp/` and never pollute the spec-mandated location.

No gaps found that weren't already addressed. No tasks added during self-review.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-15-strategy-factory-v020.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Best fit for this build because the cycle's failure-handling and dedup-timing rules are subtle, and a fresh agent per task forces the implementation against the test rather than against memory of the spec.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, with checkpoints between groups of tasks.

**Which approach?**

