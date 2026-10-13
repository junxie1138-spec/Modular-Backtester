from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRegimeParams:
    regime_window: int = 63
    pressure_window: int = 18
    reversal_threshold: float = 0.08
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapRegimeParams]):
    strategy_id = "gen_local_1791222109"

    @classmethod
    def params_type(cls) -> type[GapRegimeParams]:
        return GapRegimeParams

    @staticmethod
    def warmup_bars(params: GapRegimeParams) -> int:
        if (params.regime_window < 2 or params.pressure_window < 2
                or not 0 < params.reversal_threshold <= 1
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid gap-regime parameters")
        return int(max(params.regime_window + 2, params.pressure_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRegimeParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        prior_close = data["close"].shift(1).where(lambda x: x > 0)
        valid_open = data["open"].where(lambda x: x > 0)
        gap = (valid_open / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        intraday = (data["close"] / valid_open - 1).replace([np.inf, -np.inf], np.nan)
        # Prior sessions determine whether gaps propagate or reverse intraday.
        transmission = gap.rolling(
            params.regime_window, min_periods=params.regime_window
        ).corr(intraday).shift(1)
        transmission = transmission.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 1)
        gap_sum = gap.rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        gap_mass = gap.abs().rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        pressure = (gap_sum / gap_mass.replace(0, np.nan)).fillna(0).clip(-1, 1)
        score = (transmission * pressure).fillna(0).clip(-1, 1)
        return pd.DataFrame({
            "gap": gap.fillna(0),
            "intraday": intraday.fillna(0),
            "transmission": transmission,
            "gap_sum": gap_sum.fillna(0),
            "gap_mass": gap_mass.fillna(0),
            "pressure": pressure,
            "score": score,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRegimeParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0)
        # A neutral reading preserves the previous polarity; only reversal exits.
        polarity = pd.Series(np.select(
            [score >= params.reversal_threshold, score <= -params.reversal_threshold],
            [1.0, 0.0], default=np.nan,
        ), index=data.index).ffill().fillna(0)
        conviction = np.tanh(score.clip(lower=0) / (3 * params.reversal_threshold))
        size = params.min_size + (params.max_size - params.min_size) * conviction
        df = pd.DataFrame({"signal": polarity.astype(int), "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = GapRegimeParams()
    idx = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame(index=idx)
    readings = pd.DataFrame({"score": [0.0, 0.3, 0.02, -0.02, -0.3, 0.0]}, index=idx)
    result = GeneratedStrategy.generate_signals(bars, readings, None, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert result["size"].between(p.min_size, p.max_size).all()
    assert result["size"].iloc[2] > result["size"].iloc[3]
