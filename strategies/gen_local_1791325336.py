from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    signature_bars: int = 6
    percentile_bars: int = 60
    entry_percentile: float = 0.85
    breakout_bars: int = 10
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791325336"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if min(params.signature_bars, params.percentile_bars, params.breakout_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if params.signature_bars < 2 or not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("Require signature_bars >= 2 and 0.5 < entry_percentile < 1")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        return int(max(params.signature_bars + params.percentile_bars + 1, params.breakout_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        length = params.signature_bars
        out["path_variation"] = out["log_return"].abs().rolling(length, min_periods=length).sum()
        # Antisymmetric second-level signature for (elapsed time, log price).
        # Each time increment is 1/length; recent returns carry positive weights.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            area = area + ((length - 1 - 2 * lag) / (2.0 * length)) * out["log_return"].shift(lag)
        out["signature_area"] = area
        out["area_score"] = area.div(out["path_variation"].where(out["path_variation"] > 0))
        history = out["area_score"].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out["upper_percentile"] = history.quantile(params.entry_percentile)
        out["lower_percentile"] = history.quantile(1.0 - params.entry_percentile)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["channel_high"] = data["high"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
        out["channel_low"] = data["low"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).min()
        ready = out[["area_score", "upper_percentile", "lower_percentile", "atr", "channel_high", "channel_low"]].notna().all(axis=1) & (out["atr"] > 0)
        out["long_entry"] = ready & (close > out["channel_high"]) & (out["area_score"] > out["upper_percentile"]) & (out["area_score"] > 0)
        out["short_entry"] = ready & (close < out["channel_low"]) & (out["area_score"] < out["lower_percentile"]) & (out["area_score"] < 0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                signals[i] = position
                continue
            if position != 0:
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    # Symmetric short trail: the lowest close and stop only fall.
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entry_bar >= params.max_hold_bars:
                    position = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = i
                extreme = close[i]
                stop = close[i] - position * params.trail_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 102.0, 99.0, 100.0, 98.0, 101.0, 100.0]}, index=index)
    derived = pd.DataFrame({
        "atr": [1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
        "long_entry": [True, False, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result.data["size"] > 0).all()
    params.max_hold_bars = 2
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]


if __name__ == "__main__":
    self_check()
