from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 18
    memory_window: int = 63
    forecast_bars: int = 18
    entry_threshold: float = 0.6

    def __post_init__(self):
        if self.slope_window < 3 or self.memory_window < 3 or self.forecast_bars < 1:
            raise ValueError("Lookbacks must be positive; slope and memory windows must be at least three.")
        if not np.isfinite(self.entry_threshold) or self.entry_threshold <= 0:
            raise ValueError("entry_threshold must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791320255"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.slope_window + params.memory_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        numerator = out["log_close"] * 0.0
        for lag in range(n):
            numerator = numerator + weights[n - 1 - lag] * out["log_close"].shift(lag)
        out["slope"] = numerator / np.sum(weights ** 2)
        out["innovation"] = out["slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["autocorrelation"] = out["innovation"].rolling(
            params.memory_window, min_periods=params.memory_window
        ).corr(out["lagged_innovation"]).clip(-0.85, 0.85)
        out["volatility"] = out["return"].rolling(n, min_periods=n).std(ddof=0)
        rho = out["autocorrelation"]
        out["innovation_projection"] = (
            out["innovation"] * rho * (1.0 - rho ** params.forecast_bars) / (1.0 - rho)
        )
        out["forecast_slope"] = out["slope"] + out["innovation_projection"]
        out["slope_noise"] = out["volatility"] / np.sqrt(n)
        out["forecast_score"] = out["forecast_slope"] / out["slope_noise"].where(out["slope_noise"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        score = indicators["forecast_score"]
        eligible = score.notna() & indicators["autocorrelation"].gt(0.1)
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[eligible & score.gt(params.entry_threshold)] = 1
        direction.loc[eligible & score.lt(-params.entry_threshold)] = -1
        confirmed = direction.where(direction.eq(direction.shift(1)), 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (candidate, entry) in enumerate(zip(direction.to_numpy(), confirmed.to_numpy())):
            if position and candidate == -position:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "forecast_score": [2, 2, 0, -2, -2, -2, np.nan, 2, 2, 2],
        "autocorrelation": [0.5] * 10,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
