from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 12
    gap_noise_mult: float = 0.8
    body_efficiency: float = 0.6
    profit_target: float = 0.01
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791258789"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.lookback < 2 or params.gap_noise_mult <= 0:
            raise ValueError("lookback must be >= 2 and gap_noise_mult positive")
        if not 0 < params.body_efficiency <= 1:
            raise ValueError("body_efficiency must be in (0, 1]")
        if params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("profit_target must be positive and max_hold_bars 1 or 2")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda x: x > 0)
        out["gap"] = data["open"] / prior_close - 1.0
        out["gap_noise"] = np.sqrt(
            out["gap"].pow(2).rolling(params.lookback).mean().shift(1)
        )
        out["resistance"] = data["high"].rolling(params.lookback).max().shift(1)
        out["body_efficiency"] = (data["close"] - data["open"]) / (
            data["high"] - data["low"]
        ).where(lambda x: x > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["gap_agrees"] = (
            (out["gap"] > 0)
            & (out["gap"] >= params.gap_noise_mult * out["gap_noise"])
            & (data["open"] > out["resistance"])
        ).fillna(False)
        out["body_agrees"] = (
            out["body_efficiency"] >= params.body_efficiency
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        eligible = (
            indicators["gap_agrees"] & indicators["body_agrees"]
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if active:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_price = opens[i]
                held_bars += 1
                # Close-observed exits execute at the following open.
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
                    entry_price = np.nan
                    held_bars = 0
                else:
                    decisions[i] = 1
            elif eligible[i]:
                active = True
                entry_price = np.nan
                held_bars = 0
                decisions[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "open": [100, 100, 102, 104, 106, 108, 108],
            "high": [101, 101, 103.1, 105.2, 108.2, 109, 109],
            "low": [99, 99, 101.9, 103.8, 105.8, 107, 107],
            "close": [100, 100, 103, 105, 108, 108, 108],
            "volume": [1000] * 7,
        },
        index=pd.date_range("2020-01-01", periods=7),
    )
    strategy = GeneratedStrategy()
    for target, expected in (
        (0.01, [0, 0, 0, 0, 1, 0, 0]),
        (1.0, [0, 0, 0, 0, 1, 1, 0]),
    ):
        params = StrategyParams(
            lookback=2, gap_noise_mult=0.5,
            body_efficiency=0.5, profit_target=target,
        )
        result = strategy.generate_signals(
            data, strategy.indicators(data, params), None, params
        ).data
        assert result["signal"].tolist() == expected
        assert result.index.equals(data.index)
        assert (result["size"] > 0).all()
