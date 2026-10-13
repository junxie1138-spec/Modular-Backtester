from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    center_bars: int = 20
    atr_bars: int = 14
    dispersion_bars: int = 12
    compression_ratio: float = 0.65
    breakout_atr: float = 1.2
    stop_atr: float = 1.0
    hold_bars: int = 2

    def __post_init__(self):
        if any(not isinstance(v, int) or v < 2 for v in
               (self.center_bars, self.atr_bars, self.dispersion_bars)):
            raise ValueError("Lookbacks must be integers >= 2")
        if not isinstance(self.hold_bars, int) or not 1 <= self.hold_bars <= 2:
            raise ValueError("hold_bars must be 1 or 2")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.compression_ratio, self.breakout_atr, self.stop_atr)):
            raise ValueError("Thresholds must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790979582"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.center_bars, params.atr_bars,
                       params.dispersion_bars) + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["change"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["center"] = close.rolling(params.center_bars).mean().shift(1)
        out["dispersion"] = out["change"].rolling(
            params.dispersion_bars).std(ddof=0).shift(1)
        out["dispersion_ratio"] = out["dispersion"] / out["prior_atr"].replace(0, np.nan)
        out["upper"] = out["center"] + params.breakout_atr * out["prior_atr"]
        out["lower"] = out["center"] - params.breakout_atr * out["prior_atr"]
        out["compressed"] = out["dispersion_ratio"].le(params.compression_ratio)
        out["setup"] = np.select([
            out["compressed"] & close.gt(out["upper"]),
            out["compressed"] & close.lt(out["lower"]),
        ], [1, -1], default=0)
        # Confirm the first escape against its frozen boundary, one close later.
        out["entry"] = np.select([
            out["setup"].shift(1).eq(1) & close.gt(out["upper"].shift(1))
            & close.gt(out["previous_close"]),
            out["setup"].shift(1).eq(-1) & close.lt(out["lower"].shift(1))
            & close.lt(out["previous_close"]),
        ], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = age = 0
        high_water = low_water = stop = 0.0
        for i, (close, atr, entry) in enumerate(zip(closes, atrs, entries)):
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.stop_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirrored trailing stop for shorts; this stop only ratchets down.
                    low_water = min(low_water, close)
                    stop = min(stop, low_water + params.stop_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    position = age = 0
                    # An exit cannot also reopen a trade on the same decision bar.
                    continue
            elif entry:
                position = int(entry)
                age = 0
                high_water = low_water = close
                stop = close - position * params.stop_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        for stopped in (True, False):
            close = np.full(34, 100.0)
            close[24:28] += direction * np.array(
                [3.0, 4.0, 1.0, 0.0] if stopped else [3.0, 4.0, 5.0, 6.0])
            data = pd.DataFrame({
                "open": close,
                "high": close + 1.0, "low": close - 1.0,
                "close": close, "volume": 1000.0,
            }, index=pd.date_range("2020-01-01", periods=len(close)))
            ind = strategy.indicators(data, params)
            result = strategy.generate_signals(data, ind, ctx, params).data
            assert ind["entry"].iloc[24] == 0
            assert ind["entry"].iloc[25] == direction
            assert result["signal"].iloc[:26].eq(0).all()
            assert result["signal"].iloc[26] == direction
            assert result["signal"].iloc[27] == (0 if stopped else direction)
            assert result["signal"].iloc[28] == 0
            assert result["signal"].isin([-1, 0, 1]).all()
            assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
