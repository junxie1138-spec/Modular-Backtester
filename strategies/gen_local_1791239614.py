from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 6
    drawdown_threshold: float = 0.01
    exhaustion_ratio: float = 0.55
    upside_dominance: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791239614"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.drawdown_threshold) or not 0 < params.drawdown_threshold < 1:
            raise ValueError("drawdown_threshold must be between 0 and 1")
        if not np.isfinite(params.exhaustion_ratio) or not 0 < params.exhaustion_ratio < 1:
            raise ValueError("exhaustion_ratio must be between 0 and 1")
        if not np.isfinite(params.upside_dominance) or params.upside_dominance <= 0:
            raise ValueError("upside_dominance must be positive")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["prior_range"] = out["range"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["prior_close"] = data["close"].shift(1)
        out["down_extension"] = (out["prior_low"] - data["low"]).clip(lower=0) / out["prior_range"]
        out["up_extension"] = (data["high"] - out["prior_high"]).clip(lower=0) / out["prior_range"]
        out["prior_down_extension"] = out["down_extension"].shift(1)
        out["prior_peak"] = data["high"].rolling(params.lookback, min_periods=params.lookback).max().shift(1)
        out["drawdown"] = 1.0 - out["prior_close"] / out["prior_peak"].where(out["prior_peak"] > 0)
        out["close_change"] = data["close"] - out["prior_close"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["drawdown"] >= params.drawdown_threshold)
            & (out["prior_down_extension"] > 0)
            & (out["down_extension"] > 0)
            & (out["down_extension"] <= params.exhaustion_ratio * out["prior_down_extension"])
            & (out["up_extension"] >= params.upside_dominance * out["down_extension"])
            & (out["close_change"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, eligible in enumerate(entry):
            if exit_bar >= 0:
                if i < exit_bar:
                    raw[i] = 1
                else:
                    # Keep the scheduled exit flat even if another entry qualifies.
                    exit_bar = -1
            elif eligible:
                raw[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    for hold in (1, 2):
        params = GeneratedParams(hold_bars=hold)
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
        signal = strategy.generate_signals(data, indicators, ctx, params).data["signal"]
        expected = [0] + [int(i % (hold + 1) < hold) for i in range(7)]
        assert signal.tolist() == expected


if __name__ == "__main__":
    _self_check()
