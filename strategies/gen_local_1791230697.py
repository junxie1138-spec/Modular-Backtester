from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    atr_bars: int = 14
    streak_bars: int = 3
    gap_atr: float = 0.15
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1791230697"

    @classmethod
    def params_type(cls):
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return int(max(params.atr_bars + 2, params.streak_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        if (params.atr_bars < 1 or params.streak_bars < 1 or params.hold_bars < 1
                or not np.isfinite([params.gap_atr, params.breakeven_pct, params.trail_atr]).all()
                or params.gap_atr <= 0 or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Lookbacks, thresholds and holding period must be positive.")
        previous = data["close"].shift(1)
        gap = data["open"] - previous
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        prior_atr = atr.shift(1)
        valid = (np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
                 & (data[["open", "high", "low", "close"]] > 0).all(axis=1)
                 & prior_atr.gt(0) & previous.gt(0))
        rejected_down = valid & gap.le(-params.gap_atr * prior_atr) & data["close"].ge(previous)
        rejected_up = valid & gap.ge(params.gap_atr * prior_atr) & data["close"].le(previous)
        long_streak = rejected_down.astype(int).groupby((~rejected_down).cumsum()).cumsum()
        short_streak = rejected_up.astype(int).groupby((~rejected_up).cumsum()).cumsum()
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[long_streak.ge(params.streak_bars)] = 1
        direction.loc[short_streak.ge(params.streak_bars)] = -1
        streak = pd.concat([long_streak, short_streak], axis=1).max(axis=1)
        strength = (0.45 + 0.10 * (streak - params.streak_bars)).clip(0.45, 0.95)
        return pd.DataFrame({
            "previous_close": previous, "gap": gap, "true_range": true_range,
            "atr": atr, "prior_atr": prior_atr, "valid": valid,
            "rejected_down": rejected_down, "rejected_up": rejected_up,
            "long_streak": long_streak, "short_streak": short_streak,
            "streak": streak, "direction": direction, "strength": strength,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapRejectionParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.45, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.45).to_numpy(dtype=float)
        position, age = 0, 0
        entry, peak_profit, stop_profit, size = np.nan, 0.0, 0.0, 0.45
        armed = False
        warmup = self.warmup_bars(params)
        for i in range(n):
            usable = (np.isfinite([opens[i], closes[i], atrs[i]]).all()
                      and opens[i] > 0 and closes[i] > 0 and atrs[i] > 0)
            if not usable:
                position, entry = 0, np.nan
                continue
            if position:
                # The preceding close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                profit = position * (closes[i] - entry)
                peak_profit = max(peak_profit, profit)
                if peak_profit >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Profit-coordinate stop only rises; short price stops mirror this downward.
                    stop_profit = max(stop_profit, 0.0, peak_profit - params.trail_atr * atrs[i])
                # Close-confirmed exits are also shifted to the following bar.
                if profit <= stop_profit or age >= params.hold_bars:
                    position, entry = 0, np.nan
            elif i >= warmup and directions[i]:
                position = int(directions[i])
                entry, age, peak_profit, armed = np.nan, 0, 0.0, False
                stop_profit = -params.trail_atr * atrs[i]
                size = float(np.clip(strengths[i], 0.45, 0.95))
            signals[i] = position
            sizes[i] = size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.45).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapRejectionParams(atr_bars=3, streak_bars=3)
    j = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=j + 24, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=j)
    for direction in (1, -1):
        close = np.full(len(index), 100.0)
        close[j + 1:j + 5] = 100.0 + direction * np.array([3.0, 6.0, 4.0, 3.0])
        data = pd.DataFrame({
            "open": 100.0, "high": np.maximum(close, 100.0) + 1.0,
            "low": np.minimum(close, 100.0) - 1.0,
            "close": close, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({"atr": 1.0, "direction": 0, "strength": 0.65}, index=index)
        ind.iloc[j, ind.columns.get_loc("direction")] = direction
        ind.iloc[j + 3, ind.columns.get_loc("atr")] = 10.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].iloc[j] == 0
        assert result["signal"].iloc[j + 1:j + 5].eq(direction).all()
        assert result["signal"].iloc[j + 5] == 0
        assert result["size"].iloc[j + 1] == 0.65
        prefix = strategy.generate_signals(data.iloc[:j + 4], ind.iloc[:j + 4], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:j + 4])
        flat_data = data.copy()
        flat_data["close"] = 100.0
        expired = strategy.generate_signals(flat_data, ind, ctx, params).data
        assert expired["signal"].iloc[j + 1:j + params.hold_bars + 1].eq(direction).all()
        assert expired["signal"].iloc[j + params.hold_bars + 1] == 0
    gaps = pd.DataFrame({
        "open": 99.0, "high": 101.0, "low": 98.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    gaps.iloc[8, gaps.columns.get_loc("open")] = 100.0
    derived = strategy.indicators(gaps, params)
    assert derived["long_streak"].iloc[5:9].tolist() == [2, 3, 4, 0]
    assert derived["long_streak"].iloc[9:12].tolist() == [1, 2, 3]
    assert derived["direction"].iloc[11] == 1
    assert derived["strength"].iloc[7] > derived["strength"].iloc[6]
    assert strategy.generate_signals(gaps.iloc[:0], derived.iloc[:0], ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
