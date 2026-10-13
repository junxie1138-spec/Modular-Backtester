from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    trend_span: int = 4
    noise_window: int = 16
    innovation_threshold: float = 0.45
    slope_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = "gen_local_1791197187"

    @classmethod
    def params_type(cls):
        return TrendInnovationParams

    def warmup_bars(self, params: TrendInnovationParams) -> int:
        return int(max(5 * params.trend_span, params.noise_window + 1))

    def indicators(self, data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        if params.trend_span < 2 or params.noise_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        thresholds = np.array([params.innovation_threshold, params.slope_threshold])
        if not np.isfinite(thresholds).all() or (thresholds < 0).any():
            raise ValueError("Thresholds must be finite and nonnegative.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_price"] = np.log(close.where((close > 0) & np.isfinite(close)))
        out["log_return"] = out["log_price"].diff()
        out["smooth_level"] = out["log_price"].ewm(
            span=params.trend_span, adjust=False, min_periods=params.trend_span
        ).mean()
        out["smooth_level_twice"] = out["smooth_level"].ewm(
            span=params.trend_span, adjust=False, min_periods=params.trend_span
        ).mean()
        alpha = 2.0 / (params.trend_span + 1.0)
        out["latent_level"] = 2.0 * out["smooth_level"] - out["smooth_level_twice"]
        out["latent_slope"] = alpha / (1.0 - alpha) * (
            out["smooth_level"] - out["smooth_level_twice"]
        )
        out["forecast"] = out["latent_level"].shift(1) + out["latent_slope"].shift(1)
        out["innovation"] = out["log_price"] - out["forecast"]
        out["noise"] = out["log_return"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0)
        denominator = out["noise"].where(out["noise"] > 1e-12)
        out["normalized_innovation"] = out["innovation"] / denominator
        out["normalized_slope"] = out["latent_slope"] / denominator
        valid = (
            np.isfinite(out["normalized_innovation"])
            & np.isfinite(out["normalized_slope"])
            & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        )
        long_entry = (
            valid
            & (out["normalized_innovation"] > params.innovation_threshold)
            & (out["normalized_slope"] > params.slope_threshold)
        )
        short_entry = (
            valid
            & (out["normalized_innovation"] < -params.innovation_threshold)
            & (out["normalized_slope"] < -params.slope_threshold)
        )
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendInnovationParams,
    ) -> SignalFrame:
        direction = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, current in enumerate(direction):
            if position != 0:
                if current == -position:
                    position = 0
            elif current != 0 and current == previous:
                position = int(current)
            signals[i] = position
            previous = int(current)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_direction": [1, 1, 0, -1, -1, 0, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, TrendInnovationParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()
