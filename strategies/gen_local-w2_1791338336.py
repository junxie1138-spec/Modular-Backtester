from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRangeParams:
    range_lookback: int = 40
    compression_ratio: float = 0.85
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SeasonalRangeParams]):
    strategy_id = "gen_local-w2_1791338336"

    @classmethod
    def params_type(cls):
        return SeasonalRangeParams

    def warmup_bars(self, params: SeasonalRangeParams) -> int:
        if params.range_lookback < 2 or not 5 <= params.hold_bars <= 10:
            raise ValueError("range_lookback must be >= 2; hold_bars must be 5..10")
        if not np.isfinite(params.compression_ratio) or params.compression_ratio <= 0:
            raise ValueError("compression_ratio must be finite and positive")
        return int(params.range_lookback + 3)

    def indicators(self, data: pd.DataFrame, params: SeasonalRangeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["normalized_range"] = out["range"] / data["close"].where(data["close"] > 0)
        out["range_baseline"] = out["normalized_range"].shift(1).rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).median()
        out["range_ratio"] = (out["normalized_range"] / out["range_baseline"].where(
            out["range_baseline"] > 0
        )).replace([np.inf, -np.inf], np.nan)
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out["first_weekday"] = (data.index.dayofweek - (data.index.day - 1)) % 7
        out["third_friday_day"] = 15 + (4 - out["first_weekday"]) % 7
        out["days_after_third_friday"] = data.index.day - out["third_friday_day"]
        out["seasonal_window"] = out["days_after_third_friday"].between(3, 6)
        out["qualified"] = (
            out["seasonal_window"]
            & out["range_ratio"].le(params.compression_ratio)
            & out["range_ratio"].lt(out["range_ratio"].shift(1))
            & out["close_location"].ge(0.6)
            & out["close_location"].le(1.0)
        ).fillna(False)
        out["confirmed"] = out["qualified"] & out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalRangeParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
                continue
            if entries[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SeasonalRangeParams(range_lookback=10)
    index = pd.bdate_range("2024-01-02", periods=60)
    span = pd.Series(2.0, index=index)
    span.loc["2024-01-22"] = 0.9
    span.loc["2024-01-23"] = 0.7
    data = pd.DataFrame({
        "open": 99.9,
        "high": 100.0 + 0.2 * span,
        "low": 100.0 - 0.8 * span,
        "close": 100.0,
        "volume": 1000000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    decision = index.get_loc(pd.Timestamp("2024-01-23"))
    assert derived["qualified"].iloc[decision - 1]
    assert not derived["confirmed"].iloc[decision - 1]
    assert derived["confirmed"].iloc[decision]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params).data
    entry = decision + 1
    assert result["signal"].iloc[:entry].eq(0).all()
    assert result["signal"].iloc[entry:entry + params.hold_bars].eq(1).all()
    assert result["signal"].iloc[entry + params.hold_bars:].eq(0).all()
    assert result["size"].gt(0).all()
    pd.testing.assert_frame_equal(
        derived.iloc[:decision + 1],
        strategy.indicators(data.iloc[:decision + 1], params),
    )
