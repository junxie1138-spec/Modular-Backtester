from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_tolerance: float = 0.75
    minimum_edge: float = 0.001
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790900074"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        # Three consecutive differences plus a six-bar analog lag.
        return 10

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not np.isfinite(params.shape_tolerance) or params.shape_tolerance <= 0:
            raise ValueError("shape_tolerance must be positive and finite")
        if not np.isfinite(params.minimum_edge) or params.minimum_edge < 0:
            raise ValueError("minimum_edge must be nonnegative and finite")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        returns = close.div(close.shift(1).replace(0, np.nan)).sub(1)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        down = close.diff().lt(0)
        down_one = down.shift(1, fill_value=False)
        down_two = down.shift(2, fill_value=False)
        streak = down.astype(int) + (down & down_one).astype(int) + (down & down_one & down_two).astype(int)
        candle_range = (data["high"] - data["low"]).rolling(3, min_periods=3).mean()
        pressure = (close.shift(2) - close).div(candle_range.where(candle_range > 0))
        pressure = pressure.replace([np.inf, -np.inf], np.nan)
        out["return"] = returns.fillna(0.0)
        out["down"] = down.astype(int)
        out["streak"] = streak
        out["range_mean"] = candle_range.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["pressure"] = pressure.fillna(0.0)
        count = pd.Series(0, index=data.index, dtype=int)
        payoff = pd.Series(0.0, index=data.index)
        # Only shapes disjoint from the current shape, with known outcomes.
        for lag in (4, 5, 6):
            past_streak = streak.shift(lag)
            past_pressure = pressure.shift(lag)
            outcome = returns.shift(lag - 1)
            match = (
                streak.ge(2)
                & past_streak.eq(streak)
                & pressure.sub(past_pressure).abs().le(params.shape_tolerance)
                & outcome.notna()
            )
            out[f"past_streak_{lag}"] = past_streak.fillna(0).astype(int)
            out[f"past_pressure_{lag}"] = past_pressure.fillna(0.0)
            out[f"outcome_{lag}"] = outcome.fillna(0.0)
            out[f"match_{lag}"] = match.astype(int)
            count = count + match.astype(int)
            payoff = payoff + outcome.where(match, 0.0)
        out["analog_count"] = count
        out["analog_edge"] = payoff.div(count.replace(0, np.nan)).fillna(0.0)
        out["entry"] = (
            count.gt(0)
            & out["analog_edge"].gt(params.minimum_edge)
            & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        if not isinstance(params.hold_bars, int) or not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be an integer between 5 and 10")
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(indicators["entry"].fillna(False).to_numpy(dtype=bool)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signal[i] = 1
            elif enter:
                exit_at = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AnalogParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    close = 100.0 + np.sin(np.arange(40) * 1.2)
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    indicators = strategy.indicators(data, params)
    assert not indicators.isna().any().any()
    assert indicators.iloc[:9]["entry"].eq(False).all()
    assert indicators.iloc[:25].equals(strategy.indicators(data.iloc[:25], params))
    forced = pd.DataFrame({"entry": True}, index=index)
    signals = strategy.generate_signals(data, forced, ctx, params).data
    assert signals["signal"].iloc[:9].tolist() == [0] + [1] * 7 + [0]
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].gt(0).all()
