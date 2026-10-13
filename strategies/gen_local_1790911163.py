from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    threshold: float = 0.45


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790911163"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.lookback + 1, 4)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 4 or not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("lookback must be >= 4 and threshold must be positive and finite")
        out = pd.DataFrame(index=data.index)
        midpoint = (data["high"] + data["low"]) / 2.0
        out["log_midpoint"] = np.log(midpoint.where(midpoint > 0))
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        for name, source, length in (
            ("midpoint_slope", "log_midpoint", 4),
            ("closing_slope", "log_close", 4),
            ("latent_slope", "log_close", params.lookback),
        ):
            denominator = length * (length * length - 1) / 12.0
            out[name] = sum(
                ((length - 1) / 2.0 - lag) * out[source].shift(lag)
                for lag in range(length)
            ) / denominator
        out["slope_innovation"] = out["midpoint_slope"] - out["closing_slope"]
        out["forecast"] = 4.0 * (out["latent_slope"] + out["slope_innovation"])
        out["score"] = out["forecast"] / (2.0 * out["volatility"].where(out["volatility"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        long_entry = (
            (indicators["score"] > params.threshold)
            & (indicators["slope_innovation"] > 0)
            & (indicators["midpoint_slope"] > 0)
        )
        short_entry = (
            (indicators["score"] < -params.threshold)
            & (indicators["slope_innovation"] < 0)
            & (indicators["midpoint_slope"] < 0)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    direction = pd.Series([0, 1, 0, -1, -1, 0, 1], index=index, dtype=float)
    indicators = pd.DataFrame({
        "score": direction,
        "slope_innovation": direction,
        "midpoint_slope": direction,
    })
    indicators.iloc[0] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
