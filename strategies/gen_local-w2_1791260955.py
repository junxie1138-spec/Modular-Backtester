from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HandoffParams:
    lookback: int = 20
    gap_atr: float = 0.10
    volume_ratio: float = 1.10
    continuation_fraction: float = 0.50


class GeneratedStrategy(BaseStrategy[HandoffParams]):
    strategy_id = "gen_local-w2_1791260955"

    @classmethod
    def params_type(cls) -> type[HandoffParams]:
        return HandoffParams

    def warmup_bars(self, params: HandoffParams) -> int:
        return max(200, params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: HandoffParams) -> pd.DataFrame:
        if params.lookback < 2 or params.gap_atr <= 0 or params.volume_ratio <= 0 or params.continuation_fraction <= 0:
            raise ValueError("Lookback must be at least two and thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["prior_atr"] = true_range.rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["gap"] = data["open"] - previous_close
        out["body"] = data["close"] - data["open"]
        out["prior_body"] = out["body"].shift(1)
        average_volume = data["volume"].rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["relative_volume"] = data["volume"] / average_volume.where(average_volume > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: HandoffParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        active_volume = indicators["relative_volume"] >= params.volume_ratio
        gap = indicators["gap"]
        body = indicators["body"]
        prior_body = indicators["prior_body"]
        minimum_gap = params.gap_atr * indicators["prior_atr"]
        long_entry = (
            ready & active_volume
            & (data["close"] > indicators["ma200"])
            & (gap > minimum_gap)
            & (prior_body < 0)
            & (body >= params.continuation_fraction * gap.abs())
        ).fillna(False)
        short_entry = (
            ready & active_volume
            & (data["close"] < indicators["ma200"])
            & (gap < -minimum_gap)
            & (prior_body > 0)
            & (body <= -params.continuation_fraction * gap.abs())
        ).fillna(False)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (enter_long, enter_short) in enumerate(zip(long_entry.to_numpy(), short_entry.to_numpy())):
            # Exit only when the held direction's entry predicate flips from true to false.
            # An exit occupies a flat bar even if the opposite entry is simultaneously true.
            if position == 1:
                if not enter_long:
                    position = 0
            elif position == -1:
                if not enter_short:
                    position = 0
            elif enter_long:
                position = 1
            elif enter_short:
                position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 112.0, "low": 88.0, "close": [110.0, 110.0, 110.0, 90.0, 90.0, 90.0], "volume": 100.0}, index=index)
    indicators = pd.DataFrame({"ma200": 100.0, "prior_atr": 1.0, "gap": [1.0, 1.0, 0.0, -1.0, 0.0, 0.0], "body": [1.0, 1.0, 0.0, -1.0, 0.0, 0.0], "prior_body": [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0], "relative_volume": 2.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, HandoffParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert result.index.equals(index) and (result["size"] > 0).all()
    indicators.loc[:, :] = np.nan
    assert strategy.generate_signals(data, indicators, None, HandoffParams()).data["signal"].eq(0).all()
    assert strategy.warmup_bars(HandoffParams(lookback=250)) == 251


if __name__ == "__main__":
    _self_check()
