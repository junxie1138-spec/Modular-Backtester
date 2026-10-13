from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SqueezeParams:
    vol_window: int = 5
    atr_window: int = 14
    range_window: int = 8
    percentile_window: int = 126
    compression_q: float = 0.25
    release_q: float = 0.65
    setup_bars: int = 6
    trail_atr: float = 2.2
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[SqueezeParams]):
    strategy_id = "gen_local-w2_1790974337"

    @classmethod
    def params_type(cls):
        return SqueezeParams

    def warmup_bars(self, params: SqueezeParams) -> int:
        return int(max(params.range_window, params.atr_window + 1,
                       params.vol_window + params.percentile_window + 1))

    def indicators(self, data: pd.DataFrame, params: SqueezeParams) -> pd.DataFrame:
        if min(params.vol_window, params.atr_window, params.range_window,
               params.percentile_window, params.setup_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding periods must be positive.")
        if not 0 < params.compression_q < params.release_q < 1:
            raise ValueError("Require 0 < compression_q < release_q < 1.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite.")
        close = data["close"]
        returns = close.div(close.shift(1).replace(0, np.nan)).sub(1)
        vol = returns.rolling(params.vol_window).std(ddof=0)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        history = vol.shift(1).rolling(params.percentile_window)
        out = pd.DataFrame(index=data.index)
        out["return"] = returns
        out["vol"] = vol
        out["compression_threshold"] = history.quantile(params.compression_q)
        out["release_threshold"] = history.quantile(params.release_q)
        out["atr"] = tr.rolling(params.atr_window).mean()
        out["range_high"] = data["high"].rolling(params.range_window).max()
        out["range_low"] = data["low"].rolling(params.range_window).min()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SqueezeParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        columns = ["vol", "compression_threshold", "release_threshold",
                   "atr", "range_high", "range_low"]
        values = indicators[columns].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entered = -1
        armed = -1
        ceiling = floor = extreme = stop = 0.0
        for i, close in enumerate(closes):
            vol, compression, release, atr, high, low = values[i]
            if not np.isfinite(close) or close <= 0:
                signal[i] = side
                continue
            valid = np.isfinite(values[i]).all() and atr > 0
            if side:
                if side == 1:
                    extreme = max(extreme, close)
                    if valid:
                        stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    if valid:
                        stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entered >= params.max_hold:
                    side = 0
                    armed = -1
                signal[i] = side
                continue
            if not valid:
                armed = -1
                continue
            if armed >= 0 and i - armed > params.setup_bars:
                armed = -1
            if armed >= 0 and i > armed and vol >= release:
                if close > ceiling:
                    side = 1
                elif close < floor:
                    side = -1
                if side:
                    entered = i
                    extreme = close
                    stop = close - side * params.trail_atr * atr
                    armed = -1
            if side == 0 and armed < 0 and vol <= compression:
                armed = i
                ceiling, floor = high, low
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=11)
    close = pd.Series([100, 102, 105, 101, 100, 100, 98, 95, 99, 100, 100], index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0})
    ind = pd.DataFrame({"vol": 2.0, "compression_threshold": 0.5,
                        "release_threshold": 1.5, "atr": 1.0,
                        "range_high": close + 1, "range_low": close - 1}, index=index)
    ind.loc[index[[0, 5]], "vol"] = 0.1
    strategy = GeneratedStrategy()
    params = SqueezeParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
