from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class UnloadingParams:
    acceleration_threshold: float = 0.35
    unloading_ratio: float = 1.10
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[UnloadingParams]):
    strategy_id = "gen_local-w2_1791222555"

    @classmethod
    def params_type(cls) -> type[UnloadingParams]:
        return UnloadingParams

    def warmup_bars(self, params: UnloadingParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: UnloadingParams) -> pd.DataFrame:
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.unloading_ratio) or params.unloading_ratio < 1:
            raise ValueError("unloading_ratio must be finite and at least one")
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out = pd.DataFrame(index=data.index)
        out["daily_return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(periods=3, fill_method=None)
        out["previous_roc"] = out["roc"].shift(3)
        out["acceleration"] = out["roc"] - out["previous_roc"]
        out["noise"] = out["daily_return"].abs().rolling(6, min_periods=6).mean()
        out["scaled_acceleration"] = out["acceleration"] / (3 * out["noise"].where(out["noise"] > 0))
        out["recent_volume"] = volume.rolling(3, min_periods=3).mean()
        out["previous_volume"] = out["recent_volume"].shift(3)
        out["unloading"] = out["previous_volume"] / out["recent_volume"].where(out["recent_volume"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        unloaded = out["unloading"].ge(params.unloading_ratio)
        long_entry = unloaded & out["roc"].gt(0) & out["previous_roc"].gt(0) & out["scaled_acceleration"].ge(params.acceleration_threshold)
        short_entry = unloaded & out["roc"].lt(0) & out["previous_roc"].lt(0) & out["scaled_acceleration"].le(-params.acceleration_threshold)
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: UnloadingParams) -> SignalFrame:
        if isinstance(params.holding_bars, bool) or not isinstance(params.holding_bars, int) or not 5 <= params.holding_bars <= 10:
            raise ValueError("holding_bars must be an integer between five and ten")
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered_at = -1
        for i, candidate in enumerate(entries):
            if position:
                if i - entered_at == params.holding_bars:
                    position = 0
            elif candidate:
                position = int(candidate)
                entered_at = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    for holding in (6, 8, 10):
        count = 2 * holding + 3
        index = pd.date_range("2020-01-01", periods=count, freq="B")
        data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"entry_direction": [1] + [-1] * (count - 1)}, index=index)
        result = strategy.generate_signals(data, indicators, None, UnloadingParams(holding_bars=holding)).data
        assert result["signal"].tolist() == [0] + [1] * holding + [0] + [-1] * holding + [0]
        assert result["size"].gt(0).all()
        assert strategy.warmup_bars(UnloadingParams()) <= 10
