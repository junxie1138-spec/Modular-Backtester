from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TensionParams:
    lookback: int = 6
    gap_atr: float = 0.25
    tension_min: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TensionParams]):
    strategy_id = "gen_local_1791329114"

    @classmethod
    def params_type(cls) -> type[TensionParams]:
        return TensionParams

    def warmup_bars(self, params: TensionParams) -> int:
        return int(max(15, params.lookback + 2))

    def indicators(self, data: pd.DataFrame, params: TensionParams) -> pd.DataFrame:
        if params.lookback < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError("lookback must be >= 2 and hold_bars must be 3, 4, or 5")
        if not np.isfinite(params.gap_atr) or params.gap_atr <= 0:
            raise ValueError("gap_atr must be finite and positive")
        if not np.isfinite(params.tension_min) or params.tension_min <= 0:
            raise ValueError("tension_min must be finite and positive")
        previous_close = data["close"].shift(1)
        session_range = data["high"] - data["low"]
        true_range = pd.concat([
            session_range,
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        prior_atr = true_range.rolling(14, min_periods=14).mean().shift(1)
        prior_atr = prior_atr.where(prior_atr > 0)
        body_force = (data["close"] - data["open"]) / session_range.where(session_range > 0)
        result = pd.DataFrame({
            "previous_close": previous_close,
            "session_range": session_range,
            "true_range": true_range,
            "prior_atr": prior_atr,
            "body_force": body_force,
            "gap_atr": (data["open"] - previous_close) / prior_atr,
            "ceiling": data["close"].rolling(params.lookback, min_periods=params.lookback).max().shift(1),
            "tension": body_force.rolling(params.lookback, min_periods=params.lookback).sum().shift(1),
            "displacement_atr": data["close"].diff(params.lookback).shift(1).abs() / prior_atr,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TensionParams) -> SignalFrame:
        gap_breakout = (indicators["gap_atr"] >= params.gap_atr) & (data["open"] > indicators["ceiling"])
        stored_pressure = (indicators["tension"] >= params.tension_min) & (indicators["displacement_atr"] <= 0.5)
        candidate = (gap_breakout & stored_pressure).fillna(False)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(candidate.to_numpy(dtype=bool)):
            if i < exit_at:
                signal[i] = 1
            elif i != exit_at and enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"gap_atr": 1.0, "ceiling": 99.0, "tension": 3.0, "displacement_atr": 0.0}, index=index)
    for hold in (3, 4, 5):
        params = TensionParams(hold_bars=hold)
        result = strategy.generate_signals(data, forced, None, params).data
        expected = [0] + [0 if i % (hold + 1) == hold else 1 for i in range(len(data) - 1)]
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    params = TensionParams()
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()
    for column in forced.columns:
        missing = forced.copy()
        missing[column] = np.nan
        assert strategy.generate_signals(data, missing, None, params).data["signal"].eq(0).all()
