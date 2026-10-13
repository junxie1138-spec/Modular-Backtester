from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    frac_order: float = 0.45
    frac_window: int = 64
    norm_window: int = 40
    entry_z: float = 0.8
    atr_window: int = 14
    stop_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790978522"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params):
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_window, params.norm_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two")
        if params.entry_z <= 0 or params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Entry, stop and holding parameters must be positive")
        return int(max(200, params.frac_window + params.norm_window - 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        self.warmup_bars(params)
        close = data["close"].astype(float).where(data["close"] > 0)
        weights = np.ones(params.frac_window, dtype=float)
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # ponytail: fixed truncation; increase frac_window if tail sensitivity matters.
        frac = close.copy()
        for lag in range(1, params.frac_window):
            frac = frac + weights[lag] * close.shift(lag)
        center = frac.rolling(params.norm_window, min_periods=params.norm_window).mean()
        scale = frac.rolling(params.norm_window, min_periods=params.norm_window).std(ddof=0)
        z = (frac - center) / scale.where(scale > 0)
        previous = close.shift(1)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1)
        tr = ranges.max(axis=1, skipna=False)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ma200 = close.rolling(200, min_periods=200).mean()
        bull_entry = (close > ma200) & (z.shift(1) < -params.entry_z) & (z >= -params.entry_z)
        bear_entry = (close < ma200) & (z.shift(1) > params.entry_z) & (z <= params.entry_z)
        return pd.DataFrame({
            "frac_close": frac, "frac_center": center, "frac_scale": scale,
            "frac_z": z, "true_range": tr, "atr": atr, "ma200": ma200,
            "entry": bull_entry.astype(int) - bear_entry.astype(int),
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        self.warmup_bars(params)
        values = np.column_stack([
            data["close"].to_numpy(dtype=float),
            indicators["atr"].to_numpy(dtype=float),
            indicators["ma200"].to_numpy(dtype=float),
            indicators["frac_z"].to_numpy(dtype=float),
            indicators["entry"].to_numpy(dtype=float),
        ])
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, low_water, stop, short_stop = np.nan, np.nan, np.nan, np.nan
        for i, (close, atr, ma, z, entry) in enumerate(values):
            if not np.isfinite([close, atr, ma, z, entry]).all() or close <= 0 or atr <= 0:
                position, age = 0, 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.stop_atr * atr)
                low_water = min(low_water, close)
                short_stop = min(short_stop, low_water + params.stop_atr * atr)
                # The high-water drawdown exit applies to both directions.
                # Shorts also have a mirrored ratchet to limit adverse rebounds.
                exit_now = close <= stop or age >= params.hold_bars
                exit_now = exit_now or (position == 1 and close <= ma)
                exit_now = exit_now or (position == -1 and (close >= ma or close >= short_stop))
                if exit_now:
                    position, age = 0, 0
                    continue
            elif entry:
                position = int(entry)
                high_water = low_water = close
                stop = close - params.stop_atr * atr
                short_stop = close + params.stop_atr * atr
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), WaveParams()
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100., 100., 106., 105., 102., 101., 100., 100.], index=index)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.})
    manual = pd.DataFrame({"atr": [1., 1., 1., 5., 5., 1., 1., 1.],
                           "ma200": 90., "frac_z": 0.,
                           "entry": [0, 1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, manual, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    manual["ma200"] = 110.
    manual["entry"] = [0, -1, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, manual, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, 0, 0, 0, 0, 0]
    index = pd.date_range("2020-01-01", periods=260)
    close = pd.Series(100 + np.arange(260) * 0.1 + np.sin(np.arange(260)), index=index)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.})
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:230], strategy.indicators(data.iloc[:230], params))
    assert strategy.warmup_bars(params) >= 200
    assert np.isfinite(full.iloc[199:].to_numpy()).all()
    assert strategy.generate_signals(data.iloc[:0], full.iloc[:0], None, params).data.empty


if __name__ == "__main__":
    _self_check()
