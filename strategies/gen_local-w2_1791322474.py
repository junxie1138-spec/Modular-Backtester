from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryRepairParams:
    frac_window: int = 40
    frac_order: float = 0.45
    center_window: int = 50
    memory_window: int = 8
    drawdown_window: int = 84
    min_drawdown: float = 0.025
    crossing_level: float = 0.35
    extreme_level: float = 1.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[MemoryRepairParams]):
    strategy_id = "gen_local-w2_1791322474"

    @classmethod
    def params_type(cls) -> type[MemoryRepairParams]:
        return MemoryRepairParams

    @staticmethod
    def warmup_bars(params: MemoryRepairParams) -> int:
        windows = (params.frac_window, params.center_window,
                   params.memory_window, params.drawdown_window, params.hold_bars)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in windows):
            raise ValueError("Windows and holding duration must be positive integers")
        if params.frac_window < 2 or params.center_window < 2:
            raise ValueError("Fractional and center windows must be at least two")
        values = (params.frac_order, params.min_drawdown,
                  params.crossing_level, params.extreme_level)
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Parameters must be finite")
        if not 0.0 < params.frac_order < 1.0:
            raise ValueError("Fractional order must lie between zero and one")
        if not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("Drawdown threshold must lie between zero and one")
        if not 0.0 < params.crossing_level < params.extreme_level:
            raise ValueError("Extreme level must exceed the positive crossing level")
        return int(max(params.drawdown_window,
                       params.frac_window + params.center_window + params.memory_window))

    def indicators(self, data: pd.DataFrame, params: MemoryRepairParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        frac = close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional memory; increase frac_window for longer memory.
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            frac = frac + weight * close.shift(lag)
        out["frac_close"] = frac
        out["center"] = frac.rolling(params.center_window, min_periods=params.center_window).mean().shift(1)
        out["dispersion"] = frac.rolling(params.center_window, min_periods=params.center_window).std(ddof=0).shift(1)
        out["z"] = ((frac - out["center"]) / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["peak"] = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["recent_low"] = out["z"].shift(1).rolling(params.memory_window, min_periods=params.memory_window).min()
        out["recent_high"] = out["z"].shift(1).rolling(params.memory_window, min_periods=params.memory_window).max()
        out["previous_z"] = out["z"].shift(1)
        ready = out[["z", "previous_z", "drawdown", "recent_low", "recent_high"]].notna().all(axis=1)
        stressed = out["drawdown"].ge(params.min_drawdown)
        recovery = (out["recent_low"].le(-params.extreme_level)
                    & out["previous_z"].le(params.crossing_level)
                    & out["z"].gt(params.crossing_level))
        failure = (out["recent_high"].ge(params.extreme_level)
                   & out["previous_z"].ge(-params.crossing_level)
                   & out["z"].lt(-params.crossing_level))
        out["entry"] = np.select([ready & stressed & recovery,
                                  ready & stressed & failure], [1, -1], default=0).astype(int)
        out["strength"] = (out["z"].abs() / (1.0 + params.crossing_level)).clip(0.25, 0.95).fillna(0.25)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MemoryRepairParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.25).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side, expiry, entry_size = 0, -1, 1.0
        for i in range(len(data)):
            if side and i == expiry:
                side, entry_size = 0, 1.0
                continue
            if not side and entries[i]:
                side = int(entries[i])
                expiry = i + params.hold_bars
                entry_size = float(strengths[i])
            signals[i] = side
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = MemoryRepairParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    entries = np.zeros(25, dtype=int)
    entries[[2, 4, 7]] = [1, -1, -1]
    strengths = np.linspace(0.25, 0.95, 25)
    indicators = pd.DataFrame({"entry": entries, "strength": strengths}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1] + [0] * 14
    assert result["size"].iloc[3:6].eq(strengths[2]).all()
    assert result["size"].iloc[8:11].eq(strengths[7]).all()
    assert result["size"].gt(0.0).all()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = np.exp(5.0 + np.cumsum(np.sin(np.arange(240) / 7.0) * 0.01))
    data = pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:150], params)
    pd.testing.assert_frame_equal(full.iloc[:150], prefix)
    assert full["recent_low"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    assert not np.isinf(full.to_numpy(dtype=float)).any()


if __name__ == "__main__":
    _self_check()
