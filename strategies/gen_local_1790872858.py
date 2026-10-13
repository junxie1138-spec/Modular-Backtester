from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionParams:
    rank_bars: int = 8
    tail_rank: float = 0.75
    hold_bars: int = 7

    def __post_init__(self):
        if type(self.rank_bars) is not int or not 3 <= self.rank_bars <= 9:
            raise ValueError("rank_bars must be an integer from 3 through 9")
        if type(self.hold_bars) is not int or not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be an integer from 5 through 10")
        if not 0.5 < self.tail_rank < 1.0:
            raise ValueError("tail_rank must be strictly between 0.5 and 1")


class GeneratedStrategy(BaseStrategy[AuctionParams]):
    strategy_id = "gen_local_1790872858"

    @classmethod
    def params_type(cls):
        return AuctionParams

    def warmup_bars(self, params: AuctionParams) -> int:
        return params.rank_bars + 1

    def indicators(self, data: pd.DataFrame, params: AuctionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["intersection"] = (
            np.minimum(data["high"], out["previous_high"])
            - np.maximum(data["low"], out["previous_low"])
        ).clip(lower=0)
        out["envelope"] = (
            np.maximum(data["high"], out["previous_high"])
            - np.minimum(data["low"], out["previous_low"])
        )
        out["overlap"] = (
            out["intersection"] / out["envelope"].where(out["envelope"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        n = params.rank_bars
        # Endpoint-normalized ranks give symmetric upper and lower tails.
        out["price_rank"] = (
            data["close"].rolling(n, min_periods=n).rank(method="average") - 1.0
        ) / (n - 1.0)
        out["overlap_rank"] = (
            out["overlap"].rolling(n, min_periods=n).rank(method="average") - 1.0
        ) / (n - 1.0)
        out["standing"] = (
            (out["overlap"] >= 0.5) & (out["overlap_rank"] >= params.tail_rank)
        )
        out["traveling"] = (
            (out["overlap"] < 0.5) & (out["overlap_rank"] <= 1.0 - params.tail_rank)
        )
        out["ready"] = np.isfinite(
            out[["price_rank", "overlap_rank", "overlap"]]
        ).all(axis=1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        upper = indicators["price_rank"] >= params.tail_rank
        lower = indicators["price_rank"] <= 1.0 - params.tail_rank
        direction = upper.astype(int) - lower.astype(int)
        regime = indicators["traveling"].astype(int) - indicators["standing"].astype(int)
        df["candidate"] = (direction * regime).where(indicators["ready"], 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(df["candidate"].to_numpy()):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    # Preserve a flat exit bar; never replace the timed exit with a new entry.
                else:
                    signals[i] = position
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
                signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AuctionParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=18)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    actual = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert not actual["ready"].iloc[:8].any()
    assert not strategy.generate_signals(data, actual, ctx, params).data["signal"].any()
    forced = actual.copy()
    forced["ready"] = True
    forced["traveling"] = True
    forced["standing"] = False
    forced["price_rank"] = [1.0] + [0.0] * 17
    signal = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert signal.tolist() == [0] + [1] * 7 + [0] + [-1] * 7 + [0, -1]
    forced["traveling"], forced["standing"] = False, True
    faded = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert faded.equals(-signal)
