from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_window: int = 12
    min_degree: int = 3
    corr_window: int = 20
    corr_cutoff: float = 0.2
    atr_window: int = 14
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791260566"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.graph_window, params.corr_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.graph_window < 3 or params.corr_window < 3
                or params.atr_window < 1 or params.min_degree < 2
                or not 0 < params.corr_cutoff < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or params.hold_bars not in (1, 2)):
            raise ValueError("Invalid visibility, correlation, or exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        out["return"] = log_close.diff()
        out["autocorr"] = out["return"].rolling(params.corr_window).corr(out["return"].shift(1))
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # Horizontal visibility: intervening inverted prices lie below both endpoints.
        height = -log_close
        blocker = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        for lag in range(1, params.graph_window):
            older = height.shift(lag)
            visible = height.notna() & older.notna() & (blocker < np.minimum(height, older))
            degree += visible.astype(float)
            blocker = np.maximum(blocker, older)
        complete = height.rolling(params.graph_window).count().eq(params.graph_window)
        out["visibility_degree"] = degree.where(complete)
        out["topology_gate"] = out["visibility_degree"].ge(params.min_degree)
        out["reversal_gate"] = out["autocorr"].le(-params.corr_cutoff) & out["return"].lt(0)
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params) - 1, index=data.index)
        out["entry"] = (ready & out["topology_gate"] & out["reversal_gate"]
                        & out["atr"].gt(0) & np.isfinite(out["atr"]))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    # Initialize on the first filled session; track closes since entry.
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
                signal[i] = 1
            elif valid and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signal[i] = 1
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 97.0,
                         "close": [100.0, 100.0, 98.0, 99.0, 100.0, 101.0],
                         "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    params.hold_bars = 1
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    flat = pd.concat([data.iloc[:1]] * 40, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=40)
    assert not strategy.indicators(flat, StrategyParams())["entry"].any()


if __name__ == "__main__":
    _self_check()
