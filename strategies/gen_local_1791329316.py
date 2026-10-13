from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapQueueParams:
    atr_period: int = 14
    min_streak: int = 3
    min_gap_atr: float = 0.10
    contraction: float = 0.90
    breakeven_pct: float = 0.004
    initial_stop_atr: float = 1.25
    trail_atr: float = 0.80


class GeneratedStrategy(BaseStrategy[GapQueueParams]):
    strategy_id = "gen_local_1791329316"

    @classmethod
    def params_type(cls):
        return GapQueueParams

    def warmup_bars(self, params: GapQueueParams) -> int:
        return max(params.atr_period + 1, params.min_streak + 3)

    def indicators(self, data: pd.DataFrame, params: GapQueueParams) -> pd.DataFrame:
        if (params.atr_period < 1 or params.min_streak < 1
                or params.min_gap_atr <= 0 or not 0 < params.contraction <= 1
                or params.breakeven_pct <= 0 or params.initial_stop_atr <= 0
                or params.trail_atr <= 0):
            raise ValueError("Invalid gap-queue parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["gap"] = data["open"] - previous
        out["gap_magnitude"] = out["gap"].abs()
        valid = (data[["open", "high", "low", "close"]].gt(0).all(axis=1)
                 & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
                 & previous.gt(0) & out["atr"].gt(0))
        meaningful = valid & out["gap_magnitude"].ge(params.min_gap_atr * out["atr"].shift(1))
        up = meaningful & out["gap"].gt(0) & data["close"].gt(previous)
        down = meaningful & out["gap"].lt(0) & data["close"].lt(previous)
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        out["contracting"] = out["gap_magnitude"].le(
            params.contraction * out["gap_magnitude"].shift(1))
        out["short_setup"] = (
            valid & out["up_streak"].ge(params.min_streak)
            & out["contracting"] & data["close"].lt(data["open"]))
        out["long_setup"] = (
            valid & out["down_streak"].ge(params.min_streak)
            & out["contracting"] & data["close"].gt(data["open"]))
        out["confirmed_long"] = out["long_setup"] & out["long_setup"].shift(1, fill_value=False)
        out["confirmed_short"] = out["short_setup"] & out["short_setup"].shift(1, fill_value=False)
        out["valid"] = valid.fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapQueueParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = indicators["valid"].to_numpy(dtype=bool)
        longs = indicators["confirmed_long"].to_numpy(dtype=bool)
        shorts = indicators["confirmed_short"].to_numpy(dtype=bool)
        position = 0
        entry = np.nan
        stop = -np.inf
        peak = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if position:
                if not valid[i]:
                    position = 0
                    entry = np.nan
                    continue
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    stop = -params.initial_stop_atr * atr[i - 1]
                    peak = 0.0
                    armed = False
                    age = 0
                age += 1
                favorable = (highs[i] - entry) if position == 1 else (entry - lows[i])
                adverse = (lows[i] - entry) if position == 1 else (entry - highs[i])
                # Test the previously established stop before today's ratchet.
                exit_now = adverse <= stop
                peak = max(peak, favorable)
                armed = armed or peak >= entry * params.breakeven_pct
                if armed:
                    # Signed profit distance increases for both long and short stops.
                    stop = max(stop, 0.0, peak - params.trail_atr * atr[i])
                    close_profit = position * (float(data["close"].iloc[i]) - entry)
                    exit_now = exit_now or close_profit <= stop
                # ponytail: stop touches submit next-open exits; intrabar fills need a richer execution contract.
                if exit_now or age >= 2:
                    position = 0
                    entry = np.nan
            elif i >= self.warmup_bars(params) - 1 and valid[i]:
                if longs[i]:
                    position = 1
                    entry = np.nan
                elif shorts[i]:
                    position = -1
                    entry = np.nan
            signals[i] = position
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = GapQueueParams(atr_period=2, min_streak=2, breakeven_pct=0.5)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": [100.0] * 6 + [99.0, 99.1, 99.14, 99.3, 99.4, 99.4],
        "close": [100.0] * 6 + [99.9, 99.86, 99.8, 99.4, 99.4, 99.4],
        "volume": [1000.0] * 12,
    }, index=index)
    data["high"] = data[["open", "close"]].max(axis=1) + 0.1
    data["low"] = data[["open", "close"]].min(axis=1) - 0.1
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, None, params).data
    assert ind["confirmed_long"].iloc[8]
    assert result["signal"].iloc[8] == 0
    assert result["signal"].iloc[9:12].tolist() == [1, 1, 0]
    mirrored = data.copy()
    mirrored["open"] = 200.0 - data["open"]
    mirrored["close"] = 200.0 - data["close"]
    mirrored["high"] = 200.0 - data["low"]
    mirrored["low"] = 200.0 - data["high"]
    short = strategy.generate_signals(
        mirrored, strategy.indicators(mirrored, params), None, params).data
    assert short["signal"].iloc[9:12].tolist() == [-1, -1, 0]
    changed = data.copy()
    changed.loc[index[10]:, ["open", "high", "low", "close"]] *= 1.2
    prefix = strategy.generate_signals(
        changed, strategy.indicators(changed, params), None, params).data
    assert prefix.iloc[:10].equals(result.iloc[:10])
    fast = GapQueueParams(atr_period=2, min_streak=2, breakeven_pct=0.001, trail_atr=0.01)
    stopped = strategy.generate_signals(
        data, strategy.indicators(data, fast), None, fast).data
    assert stopped["signal"].iloc[9:11].tolist() == [1, 0]


if __name__ == "__main__":
    _self_check()
