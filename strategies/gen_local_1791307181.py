from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSupplyParams:
    volume_lookback: int = 20
    min_gap: float = 0.001
    max_volume_ratio: float = 0.85
    profit_target: float = 0.01
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapSupplyParams]):
    strategy_id = "gen_local_1791307181"

    @classmethod
    def params_type(cls):
        return GapSupplyParams

    def warmup_bars(self, params: GapSupplyParams) -> int:
        return max(4, int(params.volume_lookback) + 1)

    def indicators(self, data: pd.DataFrame, params: GapSupplyParams) -> pd.DataFrame:
        window = max(1, int(params.volume_lookback))
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = data["open"] / previous_close - 1.0
        volume = data["volume"].where(lambda s: s > 0)
        baseline = volume.shift(1).rolling(window, min_periods=window).mean()
        result = pd.DataFrame({
            "gap": gap,
            "previous_gap": gap.shift(1),
            "oldest_gap": gap.shift(2),
            "volume_ratio": volume / baseline.where(baseline > 0),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapSupplyParams) -> SignalFrame:
        if params.volume_lookback < 1 or params.min_gap <= 0:
            raise ValueError("Lookback and minimum gap must be positive")
        if not 0 < params.max_volume_ratio < 1:
            raise ValueError("Volume ratio must lie between zero and one")
        if params.profit_target <= 0 or params.holding_bars not in (1, 2):
            raise ValueError("Positive target and one- or two-bar holding period required")

        gap_agrees = (
            (indicators["oldest_gap"] > 0)
            & (indicators["previous_gap"] > indicators["oldest_gap"])
            & (indicators["gap"] > indicators["previous_gap"])
            & (indicators["gap"] >= params.min_gap)
        )
        volume_agrees = (
            (indicators["volume_ratio"] > 0)
            & (indicators["volume_ratio"] <= params.max_volume_ratio)
        )
        entries = (gap_agrees & volume_agrees).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_bar = 0

        for i in range(len(data)):
            valid = np.isfinite(opens[i]) and np.isfinite(closes[i]) and opens[i] > 0 and closes[i] > 0
            if not valid:
                active = False
                continue

            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                active = True
                entry_price = opens[i]
                entry_bar = i

            if active:
                held_bars = i - entry_bar + 1
                # Close-observed exits execute at the following open.
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                time_hit = held_bars >= params.holding_bars
                if target_hit or time_hit:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": [100.0, 102.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "gap": 0.003,
        "previous_gap": 0.002,
        "oldest_gap": 0.001,
        "volume_ratio": 0.5,
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapSupplyParams()
    result = strategy.generate_signals(sample, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    features["volume_ratio"] = 1.0
    assert strategy.generate_signals(sample, features, None, params).data["signal"].eq(0).all()
    features["volume_ratio"] = 0.5
    features["oldest_gap"] = np.nan
    assert strategy.generate_signals(sample, features, None, params).data["signal"].eq(0).all()
