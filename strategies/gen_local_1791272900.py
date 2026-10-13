from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeWorkParams:
    work_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeWorkParams]):
    strategy_id = "gen_local_1791272900"

    @classmethod
    def params_type(cls):
        return RangeWorkParams

    def warmup_bars(self, params):
        return int(max(params.work_window + params.percentile_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.work_window, params.percentile_window,
               params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and hold_bars must be positive.")
        if not 0 < params.entry_percentile < 1 or not params.stop_atr > 0:
            raise ValueError("Invalid percentile or stop distance.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data.high - data.low).where(data.high > data.low)
        previous_close = data.close.shift(1)
        out["true_range"] = pd.concat([
            data.high - data.low,
            (data.high - previous_close).abs(),
            (data.low - previous_close).abs()
        ], axis=1).max(axis=1).where(lambda s: s > 0)
        out["atr"] = out.true_range.rolling(params.atr_window).mean()
        previous_range = out["range"].shift(1)
        out["range_change"] = (
            (out["range"] - previous_range) / (out["range"] + previous_range)
        )
        out["directional_move"] = (data.close.diff() / out.true_range).clip(-1, 1)
        out["range_work"] = out.range_change * out.directional_move
        out["score"] = out.range_work.rolling(params.work_window).mean()
        out["threshold"] = (
            out.score.abs().shift(1).rolling(params.percentile_window)
            .quantile(params.entry_percentile)
        )
        out["displacement"] = data.close.diff(params.work_window)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        score = indicators.score.to_numpy()
        threshold = indicators.threshold.to_numpy()
        displacement = indicators.displacement.to_numpy()
        atr = indicators.atr.to_numpy()
        opens = data.open.to_numpy()
        closes = data.close.to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = np.nan
        frozen_atr = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry = float(opens[i])
                    stop = entry - position * params.stop_atr * frozen_atr
                age += 1
                stopped = (
                    not np.isfinite(closes[i])
                    or (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                if stopped or age >= params.hold_bars:
                    position = 0
                    entry = np.nan
                    continue
            elif (
                np.isfinite(score[i]) and np.isfinite(threshold[i])
                and np.isfinite(displacement[i]) and np.isfinite(atr[i])
                and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0
                and abs(score[i]) > threshold[i]
                and score[i] * displacement[i] > 0
            ):
                position = 1 if score[i] > 0 else -1
                frozen_atr = float(atr[i])
                entry = np.nan
                age = 0
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeWorkParams(stop_atr=2.0, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 94.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    derived = pd.DataFrame(
        {"score": 0.0, "threshold": 0.5, "displacement": 1.0, "atr": 20.0},
        index=index
    )
    derived.loc[index[1], ["score", "atr"]] = [1.0, 2.0]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for direction, breach in [(1, 95.0), (-1, 105.0)]:
        derived.loc[index[1], ["score", "displacement"]] = [direction, direction]
        data.loc[index[3], "close"] = breach
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result.signal.tolist() == [0, 0, direction, direction, 0, 0, 0, 0]
        assert result["size"].gt(0).all()
    data["close"] = 100.0
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result.signal.tolist() == [0, 0, -1, -1, -1, 0, 0, 0]
    assert strategy.warmup_bars(params) >= (
        params.work_window + params.percentile_window + 1
    )
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.generate_signals(data, actual, ctx, params).data.signal.eq(0).all()


if __name__ == "__main__":
    _self_check()
