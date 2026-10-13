from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_bars: int = 4
    noise_bars: int = 12
    drawdown_bars: int = 20
    min_drawdown: float = 0.75
    innovation_threshold: float = 0.45

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 2
               for v in (self.trend_bars, self.noise_bars, self.drawdown_bars)):
            raise ValueError("Lookbacks must be integers >= 2")
        if any(not np.isfinite(v) or v <= 0
               for v in (self.min_drawdown, self.innovation_threshold)):
            raise ValueError("Thresholds must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791245200"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_bars + 1, params.noise_bars + 2,
                       params.drawdown_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        n = params.trend_bars
        denominator = n * (n * n - 1) / 12.0
        out["level"] = out["log_close"].rolling(n, min_periods=n).mean()
        out["slope"] = sum(
            ((n - 1) / 2.0 - lag) * out["log_close"].shift(lag)
            for lag in range(n)
        ) / denominator
        out["forecast"] = (
            out["level"].shift(1) + (n + 1) / 2.0 * out["slope"].shift(1)
        )
        out["noise"] = out["return"].rolling(
            params.noise_bars, min_periods=params.noise_bars
        ).std(ddof=0).shift(1).replace(0.0, np.nan)
        out["peak"] = out["log_close"].rolling(
            params.drawdown_bars, min_periods=params.drawdown_bars
        ).max()
        out["drawdown_z"] = (out["peak"] - out["log_close"]) / out["noise"]
        # Adding today's observation to the prior fit revises its latent slope.
        gain = 6.0 / ((n + 1) * (n + 2))
        out["slope_innovation"] = gain * (out["log_close"] - out["forecast"])
        forecast_scale = np.sqrt(1.0 + 1.0 / n + ((n + 1) / 2.0) ** 2 / denominator)
        out["innovation_z"] = out["slope_innovation"] / (
            gain * out["noise"] * forecast_scale
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        eligible = indicators["drawdown_z"].ge(params.min_drawdown)
        score = indicators["innovation_z"]
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[eligible & score.ge(params.innovation_threshold)] = 1
        direction.loc[eligible & score.le(-params.innovation_threshold)] = -1
        confirmed = direction.ne(0) & direction.eq(direction.shift(1))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (candidate, ready) in enumerate(zip(direction, confirmed)):
            if position and candidate == -position:
                position = 0
            elif position == 0 and ready:
                position = int(candidate)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "innovation_z": [np.nan, 1, 1, 0, -1, -1, -1, 0, 1, 1],
        "drawdown_z": 2.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
