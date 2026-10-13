from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fast_bars: int = 3
    slow_bars: int = 9
    acceleration_lag: int = 3
    percentile_window: int = 126
    entry_quantile: float = 0.85
    atr_bars: int = 14
    trail_atr: float = 2.0
    max_hold: int = 10

    def __post_init__(self):
        if min(self.fast_bars, self.slow_bars, self.acceleration_lag,
               self.percentile_window, self.atr_bars, self.max_hold) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if self.slow_bars <= self.fast_bars:
            raise ValueError("slow_bars must exceed fast_bars.")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must lie strictly between zero and one.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790983176"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(
            params.fast_bars + params.acceleration_lag + params.percentile_window + 1,
            params.slow_bars + params.acceleration_lag + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["fast_roc"] = close / close.shift(params.fast_bars) - 1.0
        out["slow_roc"] = close / close.shift(params.slow_bars) - 1.0
        out["fast_acceleration"] = out["fast_roc"].diff(params.acceleration_lag)
        out["slow_acceleration"] = out["slow_roc"].diff(params.acceleration_lag)
        out["acceleration_threshold"] = (
            out["fast_acceleration"].shift(1)
            .rolling(params.percentile_window).quantile(params.entry_quantile)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["fast_acceleration"] > out["acceleration_threshold"])
            & (out["fast_acceleration"] > 0.0)
            & (out["fast_roc"] > 0.0)
            & (out["slow_roc"] > 0.0)
            & (out["slow_acceleration"] <= 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                held += 1
                if not valid:
                    active = False
                else:
                    # Entry fills this bar's open; only held-bar closes enter the high-water mark.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or held >= params.max_hold:
                        active = False
            elif valid and entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
            raw[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100., 100., 110., 108., 107.],
        "high": [101., 101., 111., 109., 108.],
        "low": [99., 99., 109., 107., 106.],
        "close": [100., 100., 110., 108., 107.],
        "volume": [1000.] * 5,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    forced = pd.DataFrame({
        "atr": [1., 1., 1., 10., 10.],
        "entry": [True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    short_hold = strategy.generate_signals(data, forced, ctx, WaveParams(max_hold=2)).data
    assert short_hold["signal"].tolist() == [0, 1, 1, 0, 0]
    cold = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert cold["signal"].eq(0).all()
