from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelocationParams:
    atr_days: int = 14
    gap_min_atr: float = 0.15
    relocation_level: float = 0.60
    breakeven_pct: float = 0.004
    trail_atr: float = 0.80
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RelocationParams]):
    strategy_id = "gen_local_1790854340"

    @classmethod
    def params_type(cls):
        return RelocationParams

    def warmup_bars(self, params: RelocationParams) -> int:
        return max(200, params.atr_days + 1)

    def indicators(self, data: pd.DataFrame, params: RelocationParams) -> pd.DataFrame:
        if params.atr_days < 1 or params.hold_bars not in (1, 2):
            raise ValueError("atr_days must be positive and hold_bars must be 1 or 2")
        if not 0.5 < params.relocation_level < 1.0:
            raise ValueError("relocation_level must be between 0.5 and 1")
        if not 0 < params.gap_min_atr < 0.8:
            raise ValueError("gap_min_atr must be between 0 and 0.8")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("breakeven_pct and trail_atr must be positive")

        close = data["close"]
        previous_close = close.shift(1)
        previous_low = data["low"].shift(1)
        previous_range = (data["high"] - data["low"]).shift(1).replace(0, np.nan)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["atr"] = true_range.rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["gap_atr"] = (data["open"] - previous_close) / out["atr"].shift(1).replace(0, np.nan)
        out["prior_position"] = (previous_close - previous_low) / previous_range
        out["opening_position"] = (data["open"] - previous_low) / previous_range
        out["closing_position"] = (close - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)
        level = params.relocation_level
        long_setup = (
            (close > out["ma200"])
            & out["gap_atr"].between(params.gap_min_atr, 0.8)
            & (out["prior_position"] <= 1 - level)
            & out["opening_position"].between(level, 1.0)
            & (out["closing_position"] >= level)
            & (close > data["open"])
        )
        short_setup = (
            (close < out["ma200"])
            & out["gap_atr"].between(-0.8, -params.gap_min_atr)
            & (out["prior_position"] >= level)
            & out["opening_position"].between(0.0, 1 - level)
            & (out["closing_position"] <= 1 - level)
            & (close < data["open"])
        )
        valid = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        out["setup"] = np.select([long_setup & valid, short_setup & valid], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RelocationParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        side = 0
        pending = 0
        entry = 0.0
        stop = 0.0
        armed = False
        held = 0

        for i in range(len(data)):
            if pending:
                side = pending
                pending = 0
                entry = opens[i]
                # Signed prices make the stop ratchet upward for both directions.
                stop = side * entry - params.trail_atr * atrs[i - 1]
                armed = False
                held = 0

            if side:
                held += 1
                adverse = lows[i] if side == 1 else -highs[i]
                favorable = highs[i] if side == 1 else -lows[i]
                signed_close = side * closes[i]
                exit_now = adverse <= stop
                if not exit_now:
                    if favorable - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Updates are evaluated at the close; no assumed intrabar ordering.
                        stop = max(stop, side * entry, signed_close - params.trail_atr * atrs[i])
                        exit_now = signed_close <= stop
                if exit_now or held >= params.hold_bars:
                    side = 0
                targets[i] = side
                continue

            if setups[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and i + 1 < len(data):
                pending = int(setups[i])
                targets[i] = pending

        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    bars = pd.DataFrame({
        "open": [100.0, 101.0, 103.0, 102.0],
        "high": [101.0, 102.0, 104.0, 103.0],
        "low": [99.0, 100.5, 102.0, 101.0],
        "close": [100.0, 101.5, 103.5, 102.0],
        "volume": [1000.0] * 4,
    }, index=index)
    derived = pd.DataFrame({"atr": [1.0] * 4, "setup": [1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(bars, derived, None, RelocationParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0]
    stopped = bars.copy()
    stopped.loc[index[2], "low"] = 100.0
    result = strategy.generate_signals(stopped, derived, None, RelocationParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0]
    stopped.loc[index[1], "close"] = 100.9
    result = strategy.generate_signals(stopped, derived, None, RelocationParams())
    assert result.data["signal"].tolist() == [0, 1, 0, 0]
