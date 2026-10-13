from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_bars: int = 5
    baseline_bars: int = 40
    compression_threshold: float = 0.9


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791206017"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.baseline_bars + params.compression_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.compression_bars < 1 or params.baseline_bars < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.compression_threshold) or params.compression_threshold <= 0:
            raise ValueError("Compression threshold must be finite and positive")

        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["normalized_range"] = (
            (data["high"] - data["low"]).clip(lower=0)
            / data["close"].where(data["close"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["recent_range"] = out["normalized_range"].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).mean()
        out["baseline_range"] = out["normalized_range"].shift(
            params.compression_bars
        ).rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out["range_ratio"] = (
            out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        ).replace([np.inf, -np.inf], np.nan)

        dates = data.index
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        month_start = dates.to_period("M").to_timestamp()
        previous_start = month_start - pd.offsets.MonthBegin(1)
        expiration = month_start + pd.to_timedelta(
            (4 - month_start.dayofweek) % 7 + 14, unit="D"
        )
        previous_expiration = previous_start + pd.to_timedelta(
            (4 - previous_start.dayofweek) % 7 + 14, unit="D"
        )
        latest_expiration = expiration.where(dates >= expiration, previous_expiration)
        out["days_after_expiration"] = (dates - latest_expiration).days
        out["seasonal_window"] = out["days_after_expiration"].between(0, 13)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry_condition = (
            indicators["seasonal_window"].fillna(False).astype(bool)
            & indicators["range_ratio"].lt(params.compression_threshold)
            & data["close"].gt(indicators["ma200"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # A false entry condition reverses the long signal and closes the position.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2024-06-21", periods=5, freq="B")
    data = pd.DataFrame({"close": [101.0] * 5}, index=index)
    derived = pd.DataFrame(
        {
            "ma200": [100.0] * 5,
            "range_ratio": [0.8, 0.85, 1.1, 0.8, np.nan],
            "seasonal_window": [True, True, True, False, True],
        },
        index=index,
    )
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(StrategyParams()) >= 200


if __name__ == "__main__":
    _self_check()
