from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    correlation_window: int = 42
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = "gen_local_1790927454"

    @classmethod
    def params_type(cls):
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(params.correlation_window + 4, 15))

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        if params.correlation_window < 8 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("correlation_window must be >= 8 and trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        for lag in (1, 2, 3):
            out[f"lag_{lag}"] = out["return"].shift(lag)
        for lag in (1, 3):
            out[f"corr_{lag}"] = out["return"].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out[f"lag_{lag}"]).replace([np.inf, -np.inf], np.nan)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = ((out["corr_1"] < 0.0) & (out["corr_3"] > 0.2) & (out["return"] < 0.0) & (out["lag_1"] < 0.0) & (out["lag_2"] > 0.0) & (out["atr"] > 0.0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CycleParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        stop = -np.inf
        peak = -np.inf
        armed = False
        held = 0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if not active:
                if entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                    active = True
                    entry = np.nan
                    armed = False
                    held = 0
                    raw[i] = 1
                continue
            # Record the next session's open only when that session is observed.
            if np.isnan(entry):
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    active = False
                    continue
                stop = entry - params.trail_atr * atrs[i - 1]
                peak = entry
            held += 1
            close = closes[i]
            if not np.isfinite(close) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                active = False
                continue
            # Stops are evaluated at closes; exit orders use the shifted signal.
            peak = max(peak, close)
            armed = armed or peak >= entry * 1.0075
            if armed:
                stop = max(stop, entry, peak - params.trail_atr * atrs[i])
            if close <= stop or held >= 4:
                active = False
            else:
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = CycleParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": [100.0, 101.0, 102.0, 101.5, 101.0, 101.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False], "atr": [0.25, 0.25, 0.25, 2.0, 2.0, 2.0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
