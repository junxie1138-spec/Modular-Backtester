from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    overlap_window: int = 12
    volume_window: int = 20
    atr_window: int = 14
    volume_multiple: float = 1.3
    midpoint_atr: float = 0.2
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791318515"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params):
        return int(max(params.overlap_window, params.volume_window, params.atr_window) + 1)

    def indicators(self, data, params):
        if min(params.overlap_window, params.volume_window, params.atr_window, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.volume_multiple, params.midpoint_atr, params.trail_atr)):
            raise ValueError("Thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean().replace(0, np.nan)
        out["volume_baseline"] = data["volume"].shift(1).rolling(params.volume_window).mean().replace(0, np.nan)
        out["relative_volume"] = data["volume"] / out["volume_baseline"]
        intersection = (
            pd.concat([data["high"], data["high"].shift(1)], axis=1).min(axis=1)
            - pd.concat([data["low"], data["low"].shift(1)], axis=1).max(axis=1)
        ).clip(lower=0)
        width = data["high"] - data["low"]
        smaller_width = pd.concat([width, width.shift(1)], axis=1).min(axis=1).replace(0, np.nan)
        out["overlap"] = (intersection / smaller_width).where(width.shift(1).notna())
        out["overlap_mean"] = out["overlap"].rolling(params.overlap_window).mean()
        regime = pd.Series(np.nan, index=data.index)
        regime.loc[out["overlap_mean"] >= 0.65] = 1.0
        regime.loc[out["overlap_mean"] <= 0.35] = 0.0
        out["congested"] = regime.ffill().fillna(1.0)
        out["close_advance"] = data["close"].diff() / out["atr"]
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_advance"] = out["midpoint"].diff() / out["atr"]
        out["volume_move"] = (
            (out["relative_volume"] >= params.volume_multiple + 0.5 * out["congested"])
            & (out["close_advance"] >= 0.25)
        )
        out["range_migration"] = (
            out["midpoint_advance"] >= params.midpoint_atr * (1.0 + out["congested"])
        )
        out["entry"] = (
            out["volume_move"] & out["range_migration"]
            & out[["atr", "volume_baseline", "overlap_mean", "close_advance", "midpoint_advance"]].notna().all(axis=1)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = stop = -np.inf
        age = 0
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
                active = False
                high_water = stop = -np.inf
                age = 0
                continue
            if active:
                age += 1
                # The first active bar is the fill bar, after the decision bar.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= params.max_hold:
                    active = False
                    high_water = stop = -np.inf
                    age = 0
                else:
                    signals[i] = 1
            elif entries[i]:
                active = True
                high_water = stop = -np.inf
                age = 0
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ShockwaveParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"close": [100., 102., 110., 108., 107., 106., 105., 104., 103., 102.]}, index=index)
    indicators = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 10., 10., 10., 10., 10.], "entry": [True] + [False] * 9}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0]
    assert result["size"].gt(0).all()
    prices = pd.Series(np.arange(80, dtype=float) + 100.0)
    bars = pd.DataFrame({"open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices, "volume": 1000.0})
    bars.index = pd.date_range("2020-01-01", periods=len(bars), freq="D")
    full = strategy.indicators(bars, params)
    prefix = strategy.indicators(bars.iloc[:50], params)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert not full["entry"].iloc[:strategy.warmup_bars(params) - 1].any()


if __name__ == "__main__":
    _self_check()
