from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeParams:
    ma_window: int = 20
    regime_window: int = 60
    atr_window: int = 14
    entry_z: float = 1.0
    trend_ratio: float = 1.15
    fade_ratio: float = 0.85
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.ma_window, self.regime_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < self.fade_ratio < 1 < self.trend_ratio):
            raise ValueError("Regime thresholds must straddle one.")
        if self.entry_z <= 0 or self.stop_atr <= 0 or self.hold_bars < 1:
            raise ValueError("Entry, stop and holding parameters must be positive.")


class GeneratedStrategy(BaseStrategy[RegimeParams]):
    strategy_id = "gen_local_1790977380"

    @classmethod
    def params_type(cls):
        return RegimeParams

    def warmup_bars(self, params: RegimeParams) -> int:
        return int(max(params.ma_window + 1, params.regime_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0, np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["return_1"] = close.pct_change(fill_method=None)
        out["return_5"] = close.pct_change(periods=5, fill_method=None)
        out["variance_1"] = out["return_1"].rolling(params.regime_window).var(ddof=0)
        out["variance_5"] = out["return_5"].rolling(params.regime_window).var(ddof=0)
        out["variance_ratio"] = out["variance_5"] / (5.0 * out["variance_1"].replace(0, np.nan))
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        z, previous = out["z"], out["previous_z"]
        trend = out["variance_ratio"] >= params.trend_ratio
        fade = out["variance_ratio"] <= params.fade_ratio
        long_entry = (
            (trend & (z >= params.entry_z) & (previous < params.entry_z))
            | (fade & (previous <= -params.entry_z) & (z > previous) & (z < 0))
        )
        short_entry = (
            (trend & (z <= -params.entry_z) & (previous > -params.entry_z))
            | (fade & (previous >= params.entry_z) & (z < previous) & (z > 0))
        )
        valid = np.isfinite(out[["z", "previous_z", "variance_ratio", "atr"]]).all(axis=1) & (out["atr"] > 0)
        out["candidate"] = np.select([valid & long_entry, valid & short_entry], [1, -1], default=0).astype(int)
        out["strength"] = (z.abs() / (2.0 * params.entry_z)).clip(0.25, 0.95).fillna(0.25)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RegimeParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        position, held = 0, 0
        stop, entry_atr, entry_size = np.nan, np.nan, 0.25
        for i in range(len(data)):
            if position:
                if np.isnan(stop):
                    # Entry fills on this bar; ATR was frozen at the prior close.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                held += 1
                stopped = (closes[i] < stop) if position == 1 else (closes[i] > stop)
                if stopped or held >= params.hold_bars or not np.isfinite(stop + closes[i]):
                    position = 0
                    stop = np.nan
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strength[i]):
                position = int(candidates[i])
                held = 0
                stop = np.nan
                entry_atr = atr[i]
                entry_size = float(np.clip(strength[i], 0.25, 0.95))
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RegimeParams()
    index = pd.date_range("2020-01-01", periods=5)
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": 100.0, "high": 110.0, "low": 90.0,
            "close": [100.0, 100.0 + direction * 5, 100.0 + direction, 100.0 - direction * 3, 100.0],
            "volume": 1000.0,
        }, index=index)
        features = pd.DataFrame({
            "candidate": [direction, 0, 0, 0, 0], "atr": 1.0, "strength": 0.7,
        }, index=index)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(data, features, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert np.allclose(result["size"].iloc[1:4], 0.7)
        data["close"] = 100.0
        params.hold_bars = 2
        result = strategy.generate_signals(data, features, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0]
        params.hold_bars = 8
    flat = data.copy()
    features = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, features, ctx, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
