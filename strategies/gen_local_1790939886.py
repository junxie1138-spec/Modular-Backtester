from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    frac_window: int = 48
    frac_order: float = 0.45
    impulse_bars: int = 3
    atr_window: int = 14
    gap_atr: float = 0.35
    impulse_atr: float = 0.15
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.01
    trail_atr: float = 1.8
    max_hold_bars: int = 10

    def __post_init__(self):
        if min(self.frac_window, self.impulse_bars, self.atr_window, self.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0 < self.frac_order < 1:
            raise ValueError("frac_order must be between zero and one.")
        if not all(np.isfinite(v) and v > 0 for v in (
            self.gap_atr, self.impulse_atr, self.initial_stop_atr,
            self.breakeven_pct, self.trail_atr
        )):
            raise ValueError("Thresholds and stop distances must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local_1790939886"

    @classmethod
    def params_type(cls):
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(params.frac_window + params.impulse_bars, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        # ponytail: fixed finite fractional memory; increase frac_window for longer memory.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        gap = (data["open"] - previous) / atr.shift(1).replace(0, np.nan)
        impulse = fractional.diff(params.impulse_bars) / atr.replace(0, np.nan)
        long_entry = (gap <= -params.gap_atr) & (close > previous) & (impulse > params.impulse_atr)
        short_entry = (gap >= params.gap_atr) & (close < previous) & (impulse < -params.impulse_atr)
        setup = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        return pd.DataFrame({
            "previous_close": previous, "true_range": tr, "atr": atr,
            "fractional_close": fractional, "gap_atr": gap,
            "fractional_impulse": impulse, "long_entry": long_entry.fillna(False),
            "short_entry": short_entry.fillna(False), "setup": setup
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapMemoryParams
    ) -> SignalFrame:
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry = stop = 0.0
        entered_at = 0
        armed = False
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            # Yesterday's target fills at today's open, including after an exit.
            if position == 0 and i > 0 and desired[i - 1] != 0:
                if np.isfinite(opening) and opening > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(desired[i - 1])
                    entry = opening
                    entered_at = i
                    armed = False
                    # Signed prices make the stop increase for both long and short positions.
                    stop = position * entry - params.initial_stop_atr * atr[i - 1]
            if position:
                if not np.all(np.isfinite([high, low, close, atr[i]])) or atr[i] <= 0:
                    desired[i] = position
                    continue
                adverse = low if position == 1 else -high
                if adverse <= stop or i - entered_at + 1 >= params.max_hold_bars:
                    position = 0
                    # Keep this close flat; a new entry needs a later setup.
                    continue
                signed_close = position * close
                if position * (close - entry) >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Close-confirmed ratchets take effect on the following session.
                    stop = max(stop, position * entry, signed_close - params.trail_atr * atr[i])
                desired[i] = position
            else:
                desired[i] = setup[i]
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapMemoryParams(trail_atr=1.0)
    for side in (1, -1):
        index = pd.date_range("2020-01-01", periods=5)
        closes = 100.0 + side * np.array([0, 0, 2, 0, 0], dtype=float)
        data = pd.DataFrame({
            "open": closes, "high": closes + 0.5,
            "low": closes - 0.5, "close": closes, "volume": 1000.0
        }, index=index)
        ind = pd.DataFrame({"atr": 1.0, "setup": [side, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0]
        assert (result["size"] > 0).all()
        assert strategy.indicators(data, params)["setup"].eq(0).all()
        prefix = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], None, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:3])
    assert strategy.warmup_bars(params) == 51
