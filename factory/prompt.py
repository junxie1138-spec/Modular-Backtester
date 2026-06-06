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
- Prefer vectorised pandas operations; avoid .rolling().apply() with Python
  callables where a vectorised equivalent exists.
- The signal must be mechanically computable from SPY OHLCV alone.
- DO NOT set uses_multi_symbol = True. DO NOT set uses_per_bar = True. The
  factory only supports the v0.3.0-style single-symbol contract.

ALLOWED IMPORTS - this is the EXHAUSTIVE list. Any other import is a fatal bug
that will cause the strategy to be rejected:
- `from __future__ import annotations` (recommended)
- `from dataclasses import dataclass`
- `from typing import ...` (if you need it)
- `import numpy as np`
- `import pandas as pd`
- `from backtester.core.types import SignalFrame, StrategyContext`
- `from backtester.strategies.base import BaseStrategy`

DO NOT import from `factory`, `factory.*`, `strategies`, `strategies.*`,
`tests`, `os`, `sys`, `requests`, `scipy`, `sklearn`, `statsmodels`, or any
other module not listed above. The framework only ships pandas + numpy +
the backtester contract. There are no other helpers available.

The complete imports block at the top of your strategy file should look
EXACTLY like this:

```python
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy
```

THIS IDEA'S RANDOM CONSTRAINTS:
- strategy_id (use exactly this, do not invent your own): {{strategy_id}}
- Strategy family: {{strategy_family}}
- Primary signal primitive: {{signal_primitive}}
- Target holding horizon: {{holding_horizon}}
- Direction: {{direction}} (if "long/short", you may emit -1 signals; if
  "long-only", never emit -1)
- Exit rule (the strategy MUST implement this exit mechanic): {{exit_rule}}
  Implement it inside generate_signals by driving df["signal"] to 0 when the
  exit fires - there is no config-level stop-loss block. A bar-indexed Python
  loop is acceptable for this exit computation: trailing and breakeven exits
  are path-dependent and have no clean vectorised equivalent.
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
  objective: sortino
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
