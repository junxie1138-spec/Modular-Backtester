from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetestParams:
    atr_period: int = 14
    min_gap_atr: float = 0.5
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapRetestParams]):
    strategy_id = "gen_local_1791305965"

    @classmethod
    def params_type(cls):
        return GapRetestParams

    @staticmethod
    def warmup_bars(params: GapRetestParams) -> int:
        return int(max(200, params.atr_period + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRetestParams) -> pd.DataFrame:
        if params.atr_period < 1 or params.min_gap_atr <= 0 or params.trail_atr <= 0 or params.max_hold_bars < 1:
            raise ValueError("Periods and thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_atr"] = out["gap"] / out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap_midpoint"] = (data["open"] + out["prior_close"]) / 2.0
        out["retest_base"] = out["prior_close"].shift(1)
        out["retest_top"] = data["open"].shift(1)
        out["retest_midpoint"] = out["gap_midpoint"].shift(1)
        out["setup_gap_atr"] = out["gap_atr"].shift(1)
        out["long_entry"] = (
            out["setup_gap_atr"].between(params.min_gap_atr, 2.5)
            & (data["low"].shift(1) > out["retest_base"])
            & (data["close"].shift(1) >= out["retest_top"])
            & (data["low"] > out["retest_base"])
            & (data["low"] <= out["retest_midpoint"])
            & (data["close"] >= out["retest_top"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        out["short_entry"] = (
            (-out["setup_gap_atr"]).between(params.min_gap_atr, 2.5)
            & (data["high"].shift(1) < out["retest_base"])
            & (data["close"].shift(1) <= out["retest_top"])
            & (data["high"] < out["retest_base"])
            & (data["high"] >= out["retest_midpoint"])
            & (data["close"] <= out["retest_top"])
            & (data["close"] < out["ma200"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRetestParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        watermark = np.nan
        stop = np.nan
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and np.isfinite(ma[i]) and atr[i] > 0):
                position, held, watermark, stop = 0, 0, np.nan, np.nan
                continue
            if position:
                # The first held bar is the session following the entry decision.
                held += 1
                if position == 1:
                    watermark = close[i] if np.isnan(watermark) else max(watermark, close[i])
                    candidate = watermark - params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    exit_now = close[i] <= stop or close[i] <= ma[i]
                else:
                    # Mirror the close-high trail for short positions.
                    watermark = close[i] if np.isnan(watermark) else min(watermark, close[i])
                    candidate = watermark + params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    exit_now = close[i] >= stop or close[i] >= ma[i]
                if exit_now or held >= params.max_hold_bars:
                    position, held, watermark, stop = 0, 0, np.nan, np.nan
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                held, watermark, stop = 0, np.nan, np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapRetestParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    for direction, closes, regime in [
        (1, [100, 105, 104, 101, 100], 90),
        (-1, [100, 95, 96, 99, 100], 110),
    ]:
        data = pd.DataFrame({"close": closes}, index=index)
        ind = pd.DataFrame({
            "atr": [1, 1, 10, 1, 1], "ma200": regime,
            "long_entry": [direction == 1, False, False, False, False],
            "short_entry": [direction == -1, False, False, False, False],
        }, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        timed = GeneratedStrategy.generate_signals(data, ind, None, GapRetestParams(trail_atr=2.0, max_hold_bars=2)).data
        assert timed["signal"].tolist() == [0, direction, direction, 0, 0]
    assert GeneratedStrategy.warmup_bars(GapRetestParams()) >= 200


if __name__ == "__main__":
    _self_check()
