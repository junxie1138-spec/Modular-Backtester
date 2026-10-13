from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    seasonal_bars: int = 126
    scale_bars: int = 63
    entry_accel: float = 0.35
    seasonal_floor: float = 0.05
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.roc_bars, self.seasonal_bars, self.holding_bars) < 1 or self.scale_bars < 2:
            raise ValueError("Lookbacks and holding period must be positive; scale_bars must exceed one.")
        if not np.isfinite([self.entry_accel, self.seasonal_floor]).all() or min(self.entry_accel, self.seasonal_floor) < 0:
            raise ValueError("Signal thresholds must be finite and nonnegative.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791194967"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 2 * params.roc_bars + max(params.seasonal_bars, params.scale_bars) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff(params.roc_bars).replace([np.inf, -np.inf], np.nan)
        out["scale"] = out["acceleration"].shift(1).rolling(params.scale_bars, min_periods=params.scale_bars).std().replace(0, np.nan)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        seasonal = pd.Series(np.nan, index=data.index, dtype=float)
        counts = pd.Series(np.nan, index=data.index, dtype=float)
        for day in range(5):
            mask = weekday.eq(day)
            valid = mask & out["acceleration"].notna()
            total = out["acceleration"].where(valid, 0.0).shift(1).rolling(params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            count = valid.astype(float).shift(1).rolling(params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            seasonal.loc[mask] = (total / count.replace(0, np.nan)).loc[mask]
            counts.loc[mask] = count.loc[mask]
        out["seasonal_count"] = counts
        out["seasonal_acceleration"] = seasonal.where(counts >= 8)
        out["current_strength"] = (out["acceleration"] / out["scale"]).replace([np.inf, -np.inf], np.nan)
        out["seasonal_strength"] = (out["seasonal_acceleration"] / out["scale"]).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        current = indicators["current_strength"]
        seasonal = indicators["seasonal_strength"]
        eligible = (current.gt(params.entry_accel) & seasonal.gt(params.seasonal_floor) & np.isfinite(current) & np.isfinite(seasonal)).fillna(False).to_numpy()
        conviction = (current.clip(0, 2) / 2 * seasonal.clip(0, 0.25) / 0.25).fillna(0)
        entry_sizes = (0.20 + 0.75 * conviction).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        exit_bar = -1
        active_size = 0.20
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
                sizes[i] = active_size
            elif eligible[i]:
                exit_bar = i + params.holding_bars
                active_size = float(entry_sizes[i])
                signal[i] = 1
                sizes[i] = active_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.bdate_range("2020-01-01", periods=50)
    data = pd.DataFrame({"close": np.arange(50, dtype=float) + 100}, index=index)
    derived = pd.DataFrame({"current_strength": 2.0, "seasonal_strength": 0.25}, index=index)
    derived.loc[index[1:18], "current_strength"] = -1.0
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result["signal"].iloc[:20].tolist() == [0] + [1] * 18 + [0]
    assert result["signal"].iloc[20] == 1
    assert np.allclose(result["size"].iloc[1:19], 0.95)
    assert (result["size"] > 0).all()
    assert not (result["signal"] < 0).any()
    derived[:] = np.nan
    empty = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert empty["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
