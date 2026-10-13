from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 60
    pattern_window: int = 8
    memory_window: int = 12
    min_drawdown: float = 0.03
    min_rebound: float = 0.01
    restored_patterns: int = 4
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791310626"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(params.peak_window + params.memory_window,
                       params.pattern_window + params.memory_window + 3))

    def indicators(self, data, params):
        if (min(params.peak_window, params.pattern_window, params.memory_window,
                params.hold_bars) < 1 or not 3 <= params.restored_patterns <= 6
                or not 0 < params.min_drawdown < 1
                or params.min_rebound <= 0 or params.profit_target <= 0):
            raise ValueError("Invalid recovery parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["lag2"] = close.shift(2)
        out["lag1"] = close.shift(1)
        a, b, c = out["lag2"], out["lag1"], close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & b.ne(c) & a.ne(c)
        out["pattern"] = pd.Series(np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index).where(valid)
        count_columns = []
        for k in range(6):
            name = "count_" + str(k)
            out[name] = out["pattern"].eq(k).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window).sum()
            count_columns.append(name)
        counts = out[count_columns]
        out["complexity"] = counts.gt(0).sum(axis=1).astype(float).where(
            counts.notna().all(axis=1))
        out["prior_floor"] = out["complexity"].shift(1).rolling(
            params.memory_window, min_periods=params.memory_window).min()
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = close / out["peak"] - 1.0
        out["trough_drawdown"] = out["drawdown"].rolling(
            params.memory_window, min_periods=params.memory_window).min()
        out["trough"] = close.rolling(
            params.memory_window, min_periods=params.memory_window).min()
        out["rebound"] = close / out["trough"] - 1.0
        out["strength"] = ((out["complexity"] - out["prior_floor"]) / 4.0).clip(0, 1)
        out["entry"] = (
            out["prior_floor"].le(2)
            & out["complexity"].ge(params.restored_patterns)
            & out["complexity"].shift(1).lt(params.restored_patterns)
            & out["trough_drawdown"].le(-params.min_drawdown)
            & out["drawdown"].lt(0)
            & out["rebound"].ge(params.min_rebound)
            & close.gt(out["lag1"])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        trade_size = 0.25
        for i in range(n):
            if entry_bar >= 0:
                if i == entry_bar:
                    # ponytail: observed open proxy; use actual fills if context later exposes them.
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or entry_price <= 0
                target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or target or expired:
                    entry_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = trade_size
            elif entries[i] and i + 1 < n:
                entry_bar = i + 1
                trade_size = float(0.25 + 0.70 * strength[i])
                signals[i] = 1
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    idx = pd.date_range("2020-01-01", periods=26)
    data = pd.DataFrame({k: 100.0 for k in ("open", "high", "low", "close", "volume")}, index=idx)
    ind = pd.DataFrame({"entry": [True] + [False] * 25, "strength": 0.5}, index=idx)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    assert result["size"].gt(0).all() and result["size"].iloc[1] == 0.60
    data.loc[idx[3], "close"] = 106.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[1:4].eq(1).all()
    assert result["signal"].iloc[4:].eq(0).all()
