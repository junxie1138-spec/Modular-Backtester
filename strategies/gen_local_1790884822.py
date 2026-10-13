from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParticipationParams:
    window: int = 4
    volume_window: int = 3
    coupling_threshold: float = 0.25
    cancellation_limit: float = 0.45

    def __post_init__(self):
        if not (2 <= self.window <= 5 and 2 <= self.volume_window <= 4):
            raise ValueError("Lookbacks must keep warmup at nine bars or fewer.")
        if not (0.0 < self.coupling_threshold < 1.0):
            raise ValueError("coupling_threshold must be between zero and one.")
        if not (0.0 < self.cancellation_limit < 1.0):
            raise ValueError("cancellation_limit must be between zero and one.")


class GeneratedStrategy(BaseStrategy[GapParticipationParams]):
    strategy_id = "gen_local_1790884822"

    @classmethod
    def params_type(cls):
        return GapParticipationParams

    def warmup_bars(self, params):
        return int(params.window + params.volume_window)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].where(data["volume"] >= 0)
        out["prior_volume"] = volume.shift(1).rolling(params.volume_window).mean()
        out["activity_surprise"] = (volume / out["prior_volume"].where(out["prior_volume"] > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_activity"] = out["gap"] * out["activity_surprise"]
        out["gap_energy"] = out["gap"].pow(2).rolling(params.window).mean()
        out["activity_energy"] = out["activity_surprise"].pow(2).rolling(params.window).mean()
        out["coupling_scale"] = np.sqrt(out["gap_energy"] * out["activity_energy"])
        out["mean_gap_activity"] = out["gap_activity"].rolling(params.window).mean()
        out["coupling"] = out["mean_gap_activity"] / out["coupling_scale"].where(out["coupling_scale"] > 0)
        out["mean_gap"] = out["gap"].rolling(params.window).mean()
        out["mean_absolute_gap"] = out["gap"].abs().rolling(params.window).mean()
        out["cancellation"] = out["mean_gap"].abs() / out["mean_absolute_gap"].where(out["mean_absolute_gap"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        qualified = (
            indicators["coupling"].ge(params.coupling_threshold)
            & indicators["cancellation"].le(params.cancellation_limit)
            & indicators["coupling"].notna()
            & indicators["cancellation"].notna()
        )
        df = pd.DataFrame(index=data.index)
        # Qualification reversal is the sole exit; no time-based liquidation.
        df["signal"] = qualified.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapParticipationParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert flat["signal"].eq(0).all() and flat["size"].gt(0).all()
    probes = pd.DataFrame({"coupling": [np.nan, 0.5, 0.5, -0.5, 0.5], "cancellation": [np.nan, 0.2, 0.2, 0.2, 0.2]}, index=index[:5])
    signals = strategy.generate_signals(data.iloc[:5], probes, None, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0]
    assert signals.index.equals(probes.index) and strategy.warmup_bars(params) == 7
