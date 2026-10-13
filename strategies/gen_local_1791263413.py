from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 6
    min_complexity: float = 0.75
    stretch_multiple: float = 0.75
    daily_vol_target: float = 0.006
    profit_target: float = 0.007
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791263413"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(7, min(7, max(4, int(params.complexity_window))) + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = min(7, max(4, int(params.complexity_window)))
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        a, b, c = out["return_lag2"], out["return_lag1"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Stable ties rank the older observation first.
        out["pattern"] = ((a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)).where(valid)
        out["complexity"] = 0.0
        out["novel_pattern"] = False
        for code in (0, 1, 3, 4, 6, 7):
            occurrences = out["pattern"].eq(code).astype(float).where(valid)
            out[f"count_{code}"] = occurrences.rolling(window, min_periods=window).sum()
            out[f"prior_count_{code}"] = occurrences.shift(1).rolling(window - 1, min_periods=window - 1).sum()
            out["complexity"] += (out[f"count_{code}"] / window) ** 2
            out["novel_pattern"] |= out["pattern"].eq(code) & out[f"prior_count_{code}"].eq(0)
        out["complexity"] = (1.0 - out["complexity"]) * 6.0 / 5.0
        out["volatility"] = out["return"].rolling(6, min_periods=6).std(ddof=0)
        out["stretch"] = (data["close"] / data["close"].shift(3) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["terminal_trough"] = (c < a) & (c < b)
        out["target_size"] = (max(0.0001, params.daily_vol_target) / out["volatility"].clip(lower=0.001)).clip(0.05, 0.95).fillna(0.05)
        ready = out[["complexity", "volatility", "stretch"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & out["novel_pattern"]
            & out["terminal_trough"]
            & out["complexity"].ge(params.min_complexity)
            & out["stretch"].lt(-max(0.0, params.stretch_multiple) * out["volatility"])
            & data["close"].gt(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.05}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        target_sizes = indicators["target_size"].fillna(0.05).clip(0.05, 0.95).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        holding_bars = min(2, max(1, int(params.holding_bars)))
        profit_target = max(0.0001, params.profit_target)
        entry_bar = -1
        entry_price = 0.0
        position_size = 0.05
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The prior close's entry decision fills at this open.
                    entry_price = opens[i]
                age = i - entry_bar + 1
                # A close-confirmed target schedules liquidation at the next open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + profit_target)
                if not np.isfinite(entry_price) or entry_price <= 0 or target_hit or age >= holding_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i]:
                entry_bar = i + 1
                position_size = target_sizes[i]
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    params = Params()
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].any()
    assert GeneratedStrategy.warmup_bars(params) == 9
    assert GeneratedStrategy.warmup_bars(Params(complexity_window=100)) <= 10
    synthetic = pd.DataFrame({"entry": False, "target_size": 0.4}, index=index)
    synthetic.iloc[0, 0] = True
    timed = GeneratedStrategy.generate_signals(data, synthetic, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1] + [0] * 9
    assert timed["size"].gt(0).all()
    assert timed.loc[index[1:3], "size"].eq(0.4).all()
    data.loc[index[1], "close"] = 101.0
    targeted = GeneratedStrategy.generate_signals(data, synthetic, None, params).data
    assert targeted["signal"].tolist() == [0, 1] + [0] * 10


if __name__ == "__main__":
    _self_check()
