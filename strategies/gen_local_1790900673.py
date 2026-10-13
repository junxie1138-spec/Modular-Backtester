from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalInnovationParams:
    slope_bars: int = 4
    seasonal_bars: int = 189
    forecast_bars: int = 4
    volatility_bars: int = 20
    minimum_samples: int = 12
    ridge: float = 1.0
    entry_threshold: float = 0.2


class GeneratedStrategy(BaseStrategy[SeasonalInnovationParams]):
    strategy_id = "gen_local_1790900673"

    @classmethod
    def params_type(cls) -> type[SeasonalInnovationParams]:
        return SeasonalInnovationParams

    @staticmethod
    def warmup_bars(params: SeasonalInnovationParams) -> int:
        return int(params.seasonal_bars + params.forecast_bars + max(2 * params.slope_bars + 1, params.volatility_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalInnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_bars, min_periods=params.volatility_bars).std(ddof=0)
        out["latent_slope"] = out["return"].rolling(params.slope_bars, min_periods=params.slope_bars).mean()
        out["innovation"] = ((out["latent_slope"] - out["latent_slope"].shift(params.slope_bars)) / out["volatility"].where(out["volatility"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["matured_innovation"] = out["innovation"].shift(params.forecast_bars)
        out["matured_outcome"] = (out["log_close"].diff(params.forecast_bars) / out["volatility"].shift(params.forecast_bars).where(out["volatility"].shift(params.forecast_bars) > 0)).replace([np.inf, -np.inf], np.nan)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        out["matured_weekday"] = out["weekday"].shift(params.forecast_bars)
        out["score"] = np.nan
        x = out["matured_innovation"]
        y = out["matured_outcome"]
        for weekday in range(5):
            valid = out["matured_weekday"].eq(weekday) & x.notna() & y.notna()
            count = valid.astype(float).rolling(params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            cross = (x * y).where(valid, 0.0).rolling(params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            energy = x.pow(2).where(valid, 0.0).rolling(params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            out[f"samples_{weekday}"] = count
            out[f"response_{weekday}"] = (cross / (energy + params.ridge * count).where((energy + params.ridge * count) > 0)).where(count >= params.minimum_samples)
            current = out["weekday"].eq(weekday)
            out.loc[current, "score"] = (out["innovation"] * out[f"response_{weekday}"]).loc[current]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalInnovationParams) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan)
        desired = np.select([score.gt(params.entry_threshold), score.lt(-params.entry_threshold)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SeasonalInnovationParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"score": [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, np.nan]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    mirrored = GeneratedStrategy.generate_signals(data, -indicators, None, params).data
    assert mirrored["signal"].equals(-result["signal"])
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.seasonal_bars + params.forecast_bars + params.volatility_bars + 1


if __name__ == "__main__":
    _self_check()
