from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakPressureParams:
    streak_bars: int = 3
    pressure_bars: int = 4
    pressure_threshold: float = 0.2
    atr_bars: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StreakPressureParams]):
    strategy_id = "gen_local-w2_1791324095"

    @classmethod
    def params_type(cls):
        return StreakPressureParams

    def warmup_bars(self, params: StreakPressureParams) -> int:
        if min(params.streak_bars, params.pressure_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")
        if not (0 < params.pressure_threshold < 1):
            raise ValueError("Pressure threshold must lie between zero and one.")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0):
            raise ValueError("Breakeven threshold must be finite and positive.")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("ATR multiple must be finite and positive.")
        return int(max(params.streak_bars + 1, params.pressure_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StreakPressureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["close_change"] = data["close"].diff()
        up = out["close_change"].gt(0)
        down = out["close_change"].lt(0)
        # Counts are capped: only the last streak_bars changes affect eligibility.
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum().clip(upper=params.streak_bars)
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum().clip(upper=params.streak_bars)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (
            (2 * data["close"] - data["high"] - data["low"])
            / out["range"].where(out["range"] > 0)
        ).clip(-1, 1).where(out["range"].ne(0), 0.0)
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_bars).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_bars).sum()
        out["pressure"] = out["signed_volume_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        long_entry = out["up_streak"].eq(params.streak_bars) & out["pressure"].ge(params.pressure_threshold)
        short_entry = out["down_streak"].eq(params.streak_bars) & out["pressure"].le(-params.pressure_threshold)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakPressureParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, stop, peak = 0.0, 0.0, 0.0
        armed = False
        for i, (opening, high, low, close) in enumerate(prices):
            if i + 1 < warmup or not np.isfinite([opening, high, low, close, atr[i]]).all() or min(opening, high, low, close, atr[i]) <= 0:
                side, age = 0, 0
                continue
            if side:
                if age == 0:
                    # Yesterday's close decision fills at today's observed open.
                    if i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        continue
                    entry = opening
                    peak = side * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                    armed = False
                age += 1
                # Signed-price stops ratchet upward for either direction.
                # Test the previously known stop before updating at this close.
                adverse = low if side == 1 else -high
                if adverse <= stop or age >= params.max_hold_bars:
                    side, age = 0, 0
                    continue
                peak = max(peak, side * close)
                armed = armed or side * (close - entry) >= entry * params.breakeven_pct
                if armed:
                    stop = max(stop, side * entry, peak - params.trail_atr * atr[i])
                decisions[i] = side
            else:
                side = int(candidates[i])
                age = 0
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StreakPressureParams(streak_bars=2, pressure_bars=2, atr_bars=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for side in (1, -1):
        close = 100.0 + side * np.array([0.0, 0.2, 0.4, 1.2, 0.3, 0.2])
        opening = close.copy()
        opening[3] = close[2]
        data = pd.DataFrame({
            "open": opening,
            "high": np.maximum(opening, close) + 0.1,
            "low": np.minimum(opening, close) - 0.1,
            "close": close,
            "volume": 1000.0,
        }, index=pd.date_range("2020-01-01", periods=len(close)))
        ind = strategy.indicators(data, params)
        assert ind.index.equals(data.index)
        assert ind["candidate"].iloc[2] == 0
        filtered = data.copy()
        filtered["high"] = filtered["close"] + (0.1 if side == 1 else 0.9)
        filtered["low"] = filtered["close"] - (0.9 if side == 1 else 0.1)
        assert strategy.indicators(filtered, params)["candidate"].iloc[2] == side
        # Isolate the path-dependent exit from entry filtering.
        ind["candidate"] = [0, 0, side, 0, 0, 0]
        ind["atr"] = 0.5
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, 0, side, side, 0]
        params.max_hold_bars = 1
        assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 0, 0, side, 0, 0]
        params.max_hold_bars = 2
        ind["candidate"] = 0
        assert strategy.generate_signals(data, ind, ctx, params).data["signal"].eq(0).all()
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
