from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticDrawdownParams:
    peak_window: int = 20
    vol_window: int = 20
    max_peak_age: int = 4
    entry_pressure: float = 1.6
    target_pct: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ElasticDrawdownParams]):
    strategy_id = "gen_local-w2_1791214370"

    @classmethod
    def params_type(cls):
        return ElasticDrawdownParams

    def warmup_bars(self, params):
        return int(max(params.peak_window, params.vol_window + 2))

    def indicators(self, data, params):
        if (params.peak_window < 2 or params.vol_window < 2
                or not 1 <= params.max_peak_age < params.peak_window
                or params.entry_pressure <= 0 or params.target_pct <= 0
                or params.hold_bars not in (1, 2)):
            raise ValueError("Invalid drawdown parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["returns"] = close.pct_change(fill_method=None)
        out["volatility"] = out["returns"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0).shift(1)
        out["peak"] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["peak_bar"] = out["bar_number"].where(close.eq(out["peak"])).ffill()
        out["peak_age"] = out["bar_number"] - out["peak_bar"]
        out["drawdown"] = (1.0 - close / out["peak"]).clip(lower=0)
        denominator = out["volatility"].where(out["volatility"] > 1e-8) * np.sqrt(
            out["peak_age"].where(out["peak_age"] > 0)
        )
        out["pressure"] = (out["drawdown"] / denominator).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["eligible"] = (
            out["peak_age"].between(1, params.max_peak_age)
            & out["pressure"].ge(params.entry_pressure)
            & close.notna()
        ).fillna(False)
        out["conviction"] = (
            0.25 + 0.75 * (out["pressure"] / params.entry_pressure - 1.0).clip(0, 1)
        ).fillna(0.25)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0.25).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        entry_size = 1.0
        for i in range(n):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                target_hit = (
                    valid and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.target_pct)
                )
                if not valid or target_hit or i - entry_bar + 1 >= params.hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
                    size[i] = entry_size
            elif eligible[i]:
                entry_bar = i + 1
                entry_price = np.nan
                entry_size = float(np.clip(conviction[i], 0.25, 1.0))
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        # Decisions at the close execute on the following bar; shift size together.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ElasticDrawdownParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"eligible": [False, True, False, False, False, False],
         "conviction": [0.25, 0.7, 0.25, 0.25, 0.25, 0.25]}, index=index
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].iloc[2] == 0.7
    data.loc[index[2], "close"] = 101.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    data.loc[index[2], "open"] = 120.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    flat = pd.concat([data.iloc[:1]] * 50, ignore_index=True)
    flat.index = pd.date_range("2021-01-01", periods=len(flat))
    derived = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, derived, ctx, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
