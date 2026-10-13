from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_bars: int = 63
    percentile_bars: int = 126
    depth_percentile: float = 0.75
    breakout_bars: int = 5
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790926895"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(params.peak_bars + params.percentile_bars,
                       params.peak_bars + 1, params.breakout_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if min(params.peak_bars, params.percentile_bars, params.breakout_bars,
               params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0.0 < params.depth_percentile < 1.0 or params.stop_atr <= 0.0:
            raise ValueError("Invalid percentile or ATR stop multiplier.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_bars).max()
        out["depth"] = 1.0 - close.div(out["peak"].where(out["peak"] > 0.0))
        out["depth_threshold"] = out["depth"].shift(1).rolling(
            params.percentile_bars
        ).quantile(params.depth_percentile)
        out["local_peak"] = close.shift(1).rolling(params.breakout_bars).max()
        out["momentum"] = close.div(
            close.shift(params.peak_bars).where(close.shift(params.peak_bars) > 0.0)
        ) - 1.0
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        ready = np.isfinite(indicators[
            ["depth", "depth_threshold", "local_peak", "momentum", "atr"]
        ]).all(axis=1)
        entry = (
            ready
            & (indicators["depth"] > indicators["depth_threshold"])
            & (indicators["depth"] > 0.0)
            & (data["close"] > indicators["local_peak"])
            & (indicators["momentum"] > 0.0)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target = np.zeros(len(data), dtype=int)
        active = False
        stop = np.nan
        entry_bar = -1
        for i in range(len(data)):
            # A prior close's entry decision fills at this bar's open.
            if not active and i > 0 and target[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                # Entry ATR and entry open remain frozen throughout the trade.
                if (not np.isfinite(close[i]) or close[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    target[i] = 1
            elif entry[i]:
                target[i] = 1
        df = pd.DataFrame({"signal": target, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 110.0, 107.0, 100.0, 100.0],
        "high": [101.0, 112.0, 108.0, 101.0, 101.0],
        "low": [99.0, 109.0, 106.0, 99.0, 99.0],
        "close": [100.0, 111.0, 107.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "depth": 0.1, "depth_threshold": 0.05,
        "local_peak": [99.0, 200.0, 200.0, 200.0, 200.0],
        "momentum": 0.1, "atr": [1.0, 100.0, 100.0, 100.0, 100.0],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, DrawdownParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    data["open"] = 100.0
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, ctx, DrawdownParams(hold_bars=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    ind["depth_threshold"] = np.nan
    result = strategy.generate_signals(data, ind, ctx, DrawdownParams())
    assert not result.data["signal"].any()
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
