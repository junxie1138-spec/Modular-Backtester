from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_bars: int = 12
    range_bars: int = 6
    percentile_bars: int = 126
    compression_quantile: float = 0.20
    innovation_quantile: float = 0.85
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790865473"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_bars, params.range_bars) + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.trend_bars < 2 or params.range_bars < 2 or params.percentile_bars < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and hold_bars positive.")
        if not 0 < params.compression_quantile < 1 or not 0 < params.innovation_quantile < 1:
            raise ValueError("Quantiles must lie strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        midpoint = ((data["high"] + data["low"]) / 2).where(lambda s: s > 0)
        out["log_midpoint"] = np.log(midpoint)
        length = params.trend_bars
        weights = np.arange(length, dtype=float) - (length - 1) / 2
        out["center_mean"] = out["log_midpoint"].rolling(length).mean()
        out["slope"] = sum(
            weights[j] * out["log_midpoint"].shift(length - 1 - j)
            for j in range(length)
        ) / float(np.dot(weights, weights))
        out["forecast"] = out["center_mean"].shift(1) + out["slope"].shift(1) * (length + 1) / 2
        top = data["high"].rolling(params.range_bars).max()
        bottom = data["low"].rolling(params.range_bars).min()
        out["range_width"] = np.log(top.where(top > 0) / bottom.where(bottom > 0))
        out["innovation"] = (out["log_midpoint"] - out["forecast"]) / out["range_width"].shift(1).where(out["range_width"].shift(1) > 0)
        out["compression_threshold"] = out["range_width"].shift(1).rolling(params.percentile_bars).quantile(params.compression_quantile)
        out["innovation_threshold"] = out["innovation"].shift(1).rolling(params.percentile_bars).quantile(params.innovation_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        eligible = (
            (indicators["range_width"].shift(1) <= indicators["compression_threshold"].shift(1))
            & (indicators["slope"].shift(1) > 0)
            & (indicators["innovation"] > indicators["innovation_threshold"])
            & (indicators["innovation"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    exit_bar = -1
                continue
            if eligible[bar]:
                signal[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "range_width": 0.01,
        "compression_threshold": 0.02,
        "slope": 0.001,
        "innovation": 2.0,
        "innovation_threshold": 1.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:10].eq(1).all()
    assert result["signal"].iloc[10] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
