from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 63
    segment_window: int = 3
    noise_window: int = 30
    min_depth: float = 0.025
    arrest_ratio: float = 0.25
    change_z: float = 0.7
    hold_bars: int = 8
    size_floor: float = 0.25
    size_ceiling: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790895915"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.peak_window + params.noise_window + 2 * params.segment_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(p.peak_window, min_periods=p.peak_window).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["deepening"] = out["depth"].diff()
        out["recent"] = out["deepening"].rolling(p.segment_window, min_periods=p.segment_window).mean()
        out["prior"] = out["recent"].shift(p.segment_window)
        out["noise"] = out["deepening"].rolling(p.noise_window, min_periods=p.noise_window).std(ddof=0).shift(2 * p.segment_window)
        out["change"] = (out["prior"] - out["recent"]) / out["noise"].clip(lower=1e-8)
        span = 2 * p.segment_window + 1
        out["peak_min"] = out["peak"].rolling(span, min_periods=span).min()
        out["peak_max"] = out["peak"].rolling(span, min_periods=span).max()
        out["stable_peak"] = out["peak_min"].eq(out["peak_max"]) & out["peak_min"].notna()
        out["strength"] = (out["depth"] / p.min_depth) * (out["change"] / p.change_z)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        v = indicators
        eligible = (
            v["stable_peak"]
            & v["depth"].ge(p.min_depth)
            & v["prior"].gt(0)
            & v["recent"].ge(0)
            & v["recent"].le(p.arrest_ratio * v["prior"])
            & v["change"].ge(p.change_z)
        ).fillna(False)
        eligible.iloc[:self.warmup_bars(p) - 1] = False
        trigger = eligible & ~eligible.shift(1, fill_value=False)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), p.size_floor, dtype=float)
        exit_bar = -1
        entry_size = p.size_floor
        for i in range(len(data)):
            if i < exit_bar:
                signal[i] = 1
                size[i] = entry_size
            elif i == exit_bar:
                # Reserve the scheduled exit bar for a flat target.
                continue
            elif trigger.iloc[i]:
                strength = max(0.0, float(v["strength"].iloc[i]))
                entry_size = p.size_floor + (p.size_ceiling - p.size_floor) * strength / (1.0 + strength)
                exit_bar = i + p.hold_bars
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.size_floor).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=140, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"stable_peak": False, "depth": 0.05, "prior": 0.01, "recent": 0.001, "change": 2.0, "strength": 2.0}, index=index)
    indicators.loc[index[[100, 114]], "stable_peak"] = True
    indicators.loc[index[114], "strength"] = 4.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(p))
    result = strategy.generate_signals(data, indicators, ctx, p).data
    expected = np.zeros(len(data), dtype=int)
    expected[101:109] = 1
    expected[115:123] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].iloc[101:109].nunique() == 1
    assert result["size"].iloc[115] > result["size"].iloc[101]
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, p)
    assert strategy.generate_signals(data, actual, ctx, p).data["signal"].eq(0).all()
    assert strategy.indicators(data.iloc[:110], p).equals(actual.iloc[:110])


if __name__ == "__main__":
    _self_check()
