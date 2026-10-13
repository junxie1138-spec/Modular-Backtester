from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NestingParams:
    min_streak: int = 2
    hold_bars: int = 18
    min_close_location: float = 0.75


class GeneratedStrategy(BaseStrategy[NestingParams]):
    strategy_id = "gen_local_1791242898"

    @classmethod
    def params_type(cls) -> type[NestingParams]:
        return NestingParams

    def warmup_bars(self, params: NestingParams) -> int:
        # Eight nesting comparisons plus their preceding bar and trigger lag.
        return 10

    def indicators(self, data: pd.DataFrame, params: NestingParams) -> pd.DataFrame:
        if not 1 <= params.min_streak <= 8:
            raise ValueError("min_streak must be between 1 and 8")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if not 0.0 <= params.min_close_location <= 1.0:
            raise ValueError("min_close_location must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["inside"] = (
            (data["high"] < out["previous_high"])
            & (data["low"] > out["previous_low"])
        ).fillna(False)
        positions = pd.Series(np.arange(len(data)), index=data.index, dtype=float)
        last_reset = positions.where(~out["inside"], 0.0).cummax()
        out["nesting_streak"] = (positions - last_reset).clip(upper=8).fillna(0.0)
        out["previous_streak"] = out["nesting_streak"].shift(1).fillna(0.0)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (
            (data["close"] - data["low"])
            / out["range"].where(out["range"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["previous_high"] = out["previous_high"].fillna(np.inf)
        out["previous_low"] = out["previous_low"].fillna(-np.inf)
        out["range"] = out["range"].fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: NestingParams,
    ) -> SignalFrame:
        entries = (
            (indicators["previous_streak"] >= params.min_streak)
            & (data["close"] > indicators["previous_high"])
            & (indicators["close_location"] >= params.min_close_location)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, trigger in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Exit always wins; a new trigger cannot extend the hold.
                    exit_at = -1
            elif trigger:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = NestingParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(
        {"open": 100.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=28),
    )
    data.iloc[1, :4] = [100.0, 108.0, 92.0, 101.0]
    data.iloc[2, :4] = [101.0, 106.0, 94.0, 102.0]
    data.iloc[3, :4] = [102.0, 109.0, 95.0, 108.0]
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert indicators["nesting_streak"].iloc[2] == 2
    assert np.flatnonzero(result["signal"].to_numpy()).tolist() == list(range(4, 22))
    assert result["signal"].iloc[22] == 0
    assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
