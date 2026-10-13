from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalChangeParams:
    complexity_jump: float = 0.50
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalChangeParams]):
    strategy_id = "gen_local-w2_1791197341"

    @classmethod
    def params_type(cls) -> type[OrdinalChangeParams]:
        return OrdinalChangeParams

    def warmup_bars(self, params: OrdinalChangeParams) -> int:
        # Return triples, three-pattern window, then three-bar baseline lag.
        return 9

    def indicators(self, data: pd.DataFrame, params: OrdinalChangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["old_return"] = out["return"].shift(2)
        out["middle_return"] = out["return"].shift(1)
        a, b, c = out["old_return"], out["middle_return"], out["return"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        valid &= (a != b) & (a != c) & (b != c)
        out["valid_pattern"] = valid
        out["pattern"] = ((a < b).astype(int) + 2 * (a < c).astype(int) + 4 * (b < c).astype(int)).where(valid)
        concentration = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            name = f"pattern_probability_{code}"
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(3, min_periods=3).mean()
            concentration = concentration + out[name].pow(2)
        out["complexity"] = (1.0 - concentration).clip(0.0, 1.0)
        out["baseline_complexity"] = out["complexity"].shift(3)
        out["complexity_change"] = out["complexity"] - out["baseline_complexity"]
        out["prior_displacement"] = close.shift(3).div(close.shift(6)).sub(1.0)
        out["bullish_recovery"] = (close > close.shift(1)) & (close > data["open"])
        out["entry"] = (
            out["baseline_complexity"].le(0.10)
            & out["complexity_change"].ge(params.complexity_jump)
            & out["prior_displacement"].lt(0.0)
            & out["bullish_recovery"]
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalChangeParams) -> SignalFrame:
        if not np.isfinite(params.complexity_jump) or not 0.0 < params.complexity_jump <= 2.0 / 3.0:
            raise ValueError("complexity_jump must be in (0, 2/3]")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be positive and finite")
        if not isinstance(params.max_hold, int) or isinstance(params.max_hold, bool) or not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be an integer from 3 through 5")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                # The previous close's entry fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                age += 1
                # Close-confirmed target; the exit fills at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.max_hold:
                    active = False
                    age = 0
                    continue
            elif entries[i]:
                active = True
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.loc[index[9], "entry"] = True
    strategy = GeneratedStrategy()
    params = OrdinalChangeParams()
    timed = strategy.generate_signals(data, entries, None, params).data
    assert timed["signal"].tolist() == [0] * 10 + [1] * 4 + [0] * 6
    data.loc[index[11], "close"] = 102.0
    targeted = strategy.generate_signals(data, entries, None, params).data
    assert targeted["signal"].tolist() == [0] * 10 + [1] * 2 + [0] * 8
    assert targeted["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    prefix = strategy.indicators(data.iloc[:12], params)
    assert derived.iloc[:12].equals(prefix)
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
