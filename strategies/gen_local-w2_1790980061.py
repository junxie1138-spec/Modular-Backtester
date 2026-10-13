from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 40
    shape_bars: int = 5
    archive_bars: int = 504
    hold_bars: int = 8
    neighbors: int = 12
    max_distance: float = 0.65
    min_success: float = 0.65
    entry_z: float = -0.6
    adverse_z: float = 0.75
    atr_window: int = 14
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1790980061"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.atr_window + 1, params.ma_window + params.shape_bars + params.archive_bars + params.hold_bars))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if min(p.ma_window, p.shape_bars, p.archive_bars, p.hold_bars, p.neighbors, p.atr_window) < 1:
            raise ValueError("Lookbacks and neighbor count must be positive.")
        if p.ma_window < 2 or p.shape_bars > p.hold_bars or p.archive_bars < p.shape_bars + p.hold_bars * p.neighbors:
            raise ValueError("Insufficient archive or invalid shape length.")
        if not (p.max_distance > 0 and 0 < p.min_success <= 1 and p.entry_z < 0 and p.adverse_z > 0 and p.stop_atr > 0):
            raise ValueError("Invalid thresholds.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(p.ma_window, min_periods=p.ma_window).mean()
        out["std"] = close.rolling(p.ma_window, min_periods=p.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out["tr"] = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        columns = []
        for lag in range(p.shape_bars):
            name = f"z_lag_{lag}"
            out[name] = out["z"].shift(lag)
            columns.append(name)
        shapes = out[columns].to_numpy(dtype=float)
        z = out["z"].to_numpy(dtype=float)
        success = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        count = np.zeros(len(data), dtype=int)
        offsets = np.arange(1, p.hold_bars + 1)
        # ponytail: scan a bounded archive; index shapes if multi-decade intraday data makes this costly.
        for t in range(p.ma_window + p.shape_bars + p.hold_bars, len(data)):
            if not np.isfinite(shapes[t]).all():
                continue
            latest = t - p.hold_bars - p.shape_bars
            earliest = max(p.ma_window + p.shape_bars - 2, t - p.archive_bars)
            anchors = np.arange(latest, earliest - 1, -p.hold_bars)
            if len(anchors) < p.neighbors:
                continue
            future_z = z[anchors[:, None] + offsets]
            valid = np.isfinite(shapes[anchors]).all(axis=1) & np.isfinite(future_z).all(axis=1)
            anchors = anchors[valid]
            future_z = future_z[valid]
            if len(anchors) < p.neighbors:
                continue
            distances = np.sqrt(np.mean((shapes[anchors] - shapes[t]) ** 2, axis=1))
            nearest = np.argsort(distances)[:p.neighbors]
            if distances[nearest].max() > p.max_distance:
                continue
            paths = future_z[nearest]
            origin = z[anchors[nearest], None]
            reclaim = np.where(paths >= 0.0, offsets, p.hold_bars + 1).min(axis=1)
            deepen = np.where(paths <= origin - p.adverse_z, offsets, p.hold_bars + 1).min(axis=1)
            success[t] = np.mean((reclaim <= p.hold_bars) & (reclaim < deepen))
            distance[t] = distances[nearest].mean()
            count[t] = p.neighbors
        out["analog_success"] = success
        out["analog_distance"] = distance
        out["analog_count"] = count
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        setup = ((indicators["z"] <= params.entry_z) & (indicators["analog_success"] >= params.min_success) & (indicators["atr"] > 0)).fillna(False)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if not active and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                if (np.isfinite(close[i]) and close[i] < stop) or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif confirmed[i] and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 98.0, "close": [100.0, 100.0, 110.0, 99.0, 97.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"z": [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "analog_success": 1.0, "atr": 1.0}, index=index)
    p = AnalogParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    p.hold_bars = 3
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    prices = 100.0 + 0.02 * np.arange(750) + 3.0 * np.sin(np.arange(750) / 7.0)
    history = pd.DataFrame({"open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=750))
    strategy = GeneratedStrategy()
    defaults = AnalogParams()
    full = strategy.indicators(history, defaults)
    prefix = strategy.indicators(history.iloc[:650], defaults)
    np.testing.assert_allclose(full.iloc[:650].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert (full["analog_count"] == defaults.neighbors).any()


if __name__ == "__main__":
    _self_check()
