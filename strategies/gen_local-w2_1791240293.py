from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 24
    atr_window: int = 14
    entry_threshold: float = 0.35
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791240293"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.graph_window, params.atr_window + 1))

    @staticmethod
    def indicators(data, params):
        if (params.graph_window < 3 or params.atr_window < 1
                or not 0.0 < params.entry_threshold <= 1.0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0
                or params.holding_bars < 4):
            raise ValueError("Invalid visibility strategy parameters")
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0.0))
        up_degree = pd.Series(0.0, index=data.index)
        down_degree = pd.Series(0.0, index=data.index)
        minimum_slope = pd.Series(np.inf, index=data.index)
        # Natural visibility: every intervening point lies below the edge.
        # ponytail: O(n * graph_window); use a compiled scan for very large windows.
        for lag in range(1, params.graph_window):
            slope = (log_close - log_close.shift(lag)) / lag
            visible = slope.notna() & (slope < minimum_slope)
            up_degree += (visible & (slope > 0.0)).astype(float)
            down_degree += (visible & (slope < 0.0)).astype(float)
            minimum_slope = pd.Series(
                np.minimum(minimum_slope.to_numpy(), slope.fillna(np.inf).to_numpy()),
                index=data.index,
            )
        ready = log_close.rolling(params.graph_window).count().eq(params.graph_window)
        topology = ((up_degree - down_degree) / (params.graph_window - 1)).where(ready)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        drift = log_close - log_close.shift(params.graph_window - 1)
        return pd.DataFrame({
            "log_close": log_close,
            "up_degree": up_degree.where(ready),
            "down_degree": down_degree.where(ready),
            "topology": topology,
            "drift": drift,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        score = indicators["topology"].to_numpy(dtype=float)
        drift = indicators["drift"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        active = False
        rally_regime = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and close[i] > 0.0
                     and np.isfinite(atr[i]) and atr[i] > 0.0
                     and np.isfinite(score[i]) and np.isfinite(drift[i]))
            if not valid:
                active = False
                rally_regime = False
                continue
            if score[i] >= params.entry_threshold:
                rally_regime = True
            elif score[i] <= params.entry_threshold * 0.5:
                rally_regime = False
            if active:
                age += 1
                # The first update is on the actual fill bar, after the shift.
                high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if (close[i] <= stop or age >= params.holding_bars
                        or (age >= params.holding_bars - 3 and not rally_regime)):
                    active = False
            elif rally_regime and drift[i] > 0.0:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            if active:
                signal[i] = 1
                size[i] = float(0.25 + 0.70 * np.clip(score[i], 0.0, 1.0))
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VisibilityParams()
    t = np.arange(90, dtype=float)
    close = 100.0 * np.exp(0.0001 * t * t)
    close[70:] *= 0.6
    data = pd.DataFrame({
        "open": close, "high": close * 1.003, "low": close * 0.997,
        "close": close, "volume": np.full(len(t), 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=len(t)))
    strategy = GeneratedStrategy
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    prefix = data.iloc[:65]
    earlier = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params,
    ).data
    pd.testing.assert_frame_equal(result.iloc[:65], earlier)
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].between(0.25, 0.95).all()
    assert result["signal"].iloc[:params.graph_window].eq(0).all()
    assert result["signal"].iloc[70] == 1 and result["signal"].iloc[71] == 0


if __name__ == "__main__":
    _self_check()
