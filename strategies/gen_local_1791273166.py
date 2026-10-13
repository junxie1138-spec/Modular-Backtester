from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShiftParams:
    shift_z: float = 2.0
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapShiftParams]):
    strategy_id = "gen_local_1791273166"

    @classmethod
    def params_type(cls):
        return GapShiftParams

    @staticmethod
    def warmup_bars(params: GapShiftParams) -> int:
        return 24

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapShiftParams) -> pd.DataFrame:
        if not np.isfinite(params.shift_z) or params.shift_z <= 0:
            raise ValueError("shift_z must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        previous = data["close"].shift(1).where(lambda x: x > 0)
        gap = data["open"] / previous - 1.0
        recent = gap.rolling(3, min_periods=3).mean()
        baseline = gap.shift(3).rolling(20, min_periods=20).mean()
        scale = gap.shift(3).rolling(20, min_periods=20).std(ddof=1)
        score = np.sqrt(3.0) * (recent - baseline) / scale.where(scale > 0)
        minimum = gap.rolling(3, min_periods=3).min()
        maximum = gap.rolling(3, min_periods=3).max()
        tr = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        setup = pd.Series(0, index=data.index, dtype=int)
        setup.loc[(score >= params.shift_z) & (minimum > 0)] = 1
        setup.loc[(score <= -params.shift_z) & (maximum < 0)] = -1
        return pd.DataFrame(
            {"gap": gap, "recent_mean": recent, "baseline_mean": baseline,
             "baseline_std": scale, "score": score, "minimum_gap": minimum,
             "maximum_gap": maximum, "true_range": tr, "atr": atr,
             "setup": setup},
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShiftParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        pending = position = age = 0
        entry = best = stop = 0.0
        armed = True
        breakeven = False
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite(prices[i]).all() and min(prices[i]) > 0
            if not valid:
                pending = position = 0
                continue
            if np.isfinite(scores[i]) and abs(scores[i]) < 0.5:
                armed = True
            if pending:
                if i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    pending = 0
                    continue
                position, pending = pending, 0
                # Entry uses this open only after the previous close scheduled it.
                entry = position * opening
                best = entry
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                breakeven = False
            if position:
                age += 1
                adverse = low if position == 1 else -high
                favorable = high if position == 1 else -low
                # Test the previously established stop before tightening it.
                # A detected breach schedules liquidation at the next open.
                if adverse <= stop or age >= 4:
                    position = 0
                    continue
                best = max(best, favorable)
                if best - entry >= 0.006 * abs(entry):
                    breakeven = True
                if breakeven:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        # Signed coordinates make both directions ratchet upward.
                        stop = max(stop, best - params.trail_atr * atr[i])
                raw[i] = position
            elif armed and setups[i] and np.isfinite(scores[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(setups[i])
                armed = False
                raw[i] = pending
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapShiftParams()
    index = pd.date_range("2020-01-01", periods=9)
    for direction in (1, -1):
        for activate in (False, True):
            data = pd.DataFrame(
                {"open": 100.0, "high": 100.4, "low": 99.5,
                 "close": 100.0, "volume": 1000.0}, index=index,
            )
            if activate:
                data.iloc[1, data.columns.get_loc("high")] = 102.0
                data.iloc[1, data.columns.get_loc("close")] = 101.0
            if direction == -1:
                high, low = data["high"].copy(), data["low"].copy()
                data["open"] = 200.0 - data["open"]
                data["close"] = 200.0 - data["close"]
                data["high"], data["low"] = 200.0 - low, 200.0 - high
            ind = pd.DataFrame(
                {"atr": 2.0, "score": 0.0, "setup": 0}, index=index,
            )
            ind.iloc[0, ind.columns.get_loc("score")] = 3.0 * direction
            ind.iloc[0, ind.columns.get_loc("setup")] = direction
            result = GeneratedStrategy.generate_signals(data, ind, None, params).data
            expected = np.zeros(9, dtype=int)
            expected[1:3 if activate else 5] = direction
            assert np.array_equal(result["signal"].to_numpy(), expected)
            assert (result["size"] > 0).all()
    t = np.arange(60, dtype=float)
    close = 100.0 + 0.1 * t + np.sin(t)
    data = pd.DataFrame(
        {"open": close + 0.2 * np.cos(t), "high": close + 1.0,
         "low": close - 1.0, "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=60),
    )
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:45], params)
    assert np.allclose(full.iloc[:45], prefix, equal_nan=True)
    full_signal = GeneratedStrategy.generate_signals(data, full, None, params).data
    prefix_signal = GeneratedStrategy.generate_signals(data.iloc[:45], prefix, None, params).data
    assert np.array_equal(full_signal["signal"].iloc[:45], prefix_signal["signal"])


if __name__ == "__main__":
    _self_check()
