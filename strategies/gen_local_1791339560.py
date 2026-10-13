from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    history_bars: int = 126
    threshold: float = 0.20


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = "gen_local_1791339560"

    @classmethod
    def params_type(cls) -> type[SeasonalSlopeParams]:
        return SeasonalSlopeParams

    def warmup_bars(self, params: SeasonalSlopeParams) -> int:
        return max(20, int(params.history_bars)) + 6

    def indicators(self, data: pd.DataFrame, params: SeasonalSlopeParams) -> pd.DataFrame:
        window = max(20, int(params.history_bars))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["latent_slope"] = out["log_return"].rolling(5, min_periods=5).mean()
        out["slope_innovation"] = out["latent_slope"].diff()
        out["innovation_scale"] = out["slope_innovation"].rolling(window, min_periods=window).std(ddof=0)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekday"] = weekday
        valid = out["slope_innovation"].notna()
        for day in range(5):
            selected = weekday.eq(day) & valid
            count = selected.astype(float).rolling(window, min_periods=window).sum()
            total = out["slope_innovation"].where(selected, 0.0).rolling(window, min_periods=window).sum()
            out[f"weekday_count_{day}"] = count
            out[f"weekday_innovation_{day}"] = total.div(count.where(count > 0))
        forecast = pd.Series(0.0, index=data.index)
        for offset in range(1, 8):
            # Calendar projection uses weekdays only, never future data rows.
            next_day = (weekday + offset) % 5
            expected = pd.Series(0.0, index=data.index)
            for day in range(5):
                expected = expected + out[f"weekday_innovation_{day}"].where(next_day.eq(day), 0.0)
            forecast = forecast + (8 - offset) * expected
        out["seasonal_displacement"] = forecast
        out["score"] = forecast.div(out["innovation_scale"].where(out["innovation_scale"] > 0) * np.sqrt(140.0))
        out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalSlopeParams) -> SignalFrame:
        threshold = max(0.0, float(params.threshold))
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        entry = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite entry condition closes the position for this decision bar.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"score": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, SeasonalSlopeParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result.data["size"].gt(0).all()
