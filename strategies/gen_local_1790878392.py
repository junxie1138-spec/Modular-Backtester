from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    compression_ratio: float = 0.95
    density_ratio: float = 1.0
    innovation_floor: float = 0.08


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1790878392"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 5

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        center = (data["high"] + data["low"] + 2.0 * data["close"]) / 4.0
        out["log_center"] = np.log(center.where(center > 0.0))
        out["log_range"] = np.log(
            data["high"].where(data["high"] > 0.0)
            / data["low"].where(data["low"] > 0.0)
        ).where(data["high"] > data["low"])
        out["prior_range"] = out["log_range"].rolling(3, min_periods=3).mean().shift(1)
        # The three-point regression slope is half the endpoint difference.
        out["slope"] = (out["log_center"] - out["log_center"].shift(2)) / 2.0
        out["predicted_slope"] = 2.0 * out["slope"].shift(1) - out["slope"].shift(2)
        out["innovation"] = (
            out["slope"] - out["predicted_slope"]
        ) / out["prior_range"].where(out["prior_range"] > 0.0)
        out["compression"] = out["log_range"] / out["prior_range"]
        out["density"] = data["volume"].where(data["volume"] > 0.0) / out["log_range"]
        out["prior_density"] = out["density"].rolling(3, min_periods=3).mean().shift(1)
        out["density_ratio"] = out["density"] / out["prior_density"].where(
            out["prior_density"] > 0.0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["compression"] <= params.compression_ratio)
            & (indicators["density_ratio"] >= params.density_ratio)
            & (indicators["innovation"].abs() > params.innovation_floor)
        ).fillna(False)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[eligible & (indicators["innovation"] > 0.0)] = 1
        entry.loc[eligible & (indicators["innovation"] < 0.0)] = -1
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, desired in enumerate(entry.to_numpy()):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    probe = pd.DataFrame(
        {"compression": 0.5, "density_ratio": 2.0,
         "innovation": [np.nan, 0.2, 0.0, -0.2, -0.2, 0.0, 0.2, 0.2]},
        index=index,
    )
    result = strategy.generate_signals(data, probe, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert not np.isinf(actual.to_numpy()).any()
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
