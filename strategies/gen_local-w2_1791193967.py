from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 20
    burden_window: int = 5
    percentile_window: int = 126
    tail_quantile: float = 0.70
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791193967"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params):
        return int(max(
            params.peak_window + params.burden_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data, params):
        if min(params.peak_window, params.burden_window,
               params.percentile_window, params.atr_window) < 2:
            raise ValueError("Window lengths must be at least two.")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must be between 0.5 and 1.")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Stop parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(params.peak_window).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].where(out["peak"] > 0)).clip(lower=0)
        out["burden"] = out["depth"].rolling(params.burden_window).mean()
        for name in ("depth", "burden"):
            history = out[name].shift(1).rolling(params.percentile_window)
            out[name + "_upper"] = history.quantile(params.tail_quantile)
            out[name + "_lower"] = history.quantile(1.0 - params.tail_quantile)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        long_entry = (
            (indicators["depth"] > indicators["depth_upper"])
            & (indicators["burden"] < indicators["burden_lower"])
        )
        short_entry = (
            (indicators["depth"] < indicators["depth_lower"])
            & (indicators["burden"] > indicators["burden_upper"])
        )
        ready = indicators[[
            "depth", "burden", "depth_upper", "depth_lower",
            "burden_upper", "burden_lower", "atr",
        ]].notna().all(axis=1) & (indicators["atr"] > 0)
        entries = np.select(
            [ready & long_entry, ready & short_entry], [1, -1], default=0
        )
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = 0.0
        floor = 0.0
        armed = False
        held = 0
        for i in range(len(data)):
            opened, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and min(prices[i]) > 0
            if not valid:
                side = 0
                continue
            # Yesterday's decision fills at today's open, never yesterday's close.
            if side == 0 and i > 0 and raw[i - 1] != 0:
                side = int(raw[i - 1])
                entry = opened
                held = 0
                armed = False
                floor = -params.trail_atr * atr[i - 1] / entry
            if side:
                held += 1
                # Signed return makes the protective floor increase for both sides.
                # The short price stop correspondingly decreases.
                worst = side * ((low if side == 1 else high) / entry - 1.0)
                stopped = worst <= floor
                gain = side * (close / entry - 1.0)
                if gain >= params.breakeven_pct:
                    armed = True
                if armed:
                    floor = max(floor, 0.0)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        floor = max(floor, gain - params.trail_atr * atr[i] / entry)
                    stopped = stopped or gain <= floor
                # Exits are close decisions executed at the next open.
                if stopped or held >= 2:
                    side = 0
                raw[i] = side
                # Do not reopen on the same bar as an exit.
            else:
                raw[i] = int(entries[i])
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = DrawdownParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({
        "open": [100.0] * 5, "high": [100.2] * 5,
        "low": [99.8] * 5, "close": [100.0] * 5,
        "volume": [1000.0] * 5,
    }, index=index)
    features = pd.DataFrame({
        "depth": [0.3, 0.15, 0.15, 0.15, 0.15],
        "burden": [0.05, 0.15, 0.15, 0.15, 0.15],
        "depth_upper": 0.2, "depth_lower": 0.1,
        "burden_upper": 0.2, "burden_lower": 0.1, "atr": 1.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(prices, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    prices.loc[index[1], "low"] = 98.0
    result = strategy.generate_signals(prices, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    prices.loc[index[1], "low"] = 99.8
    features.loc[index[0], ["depth", "burden"]] = [0.05, 0.3]
    result = strategy.generate_signals(prices, features, ctx, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0]
    prices.loc[index[1], "high"] = 102.0
    result = strategy.generate_signals(prices, features, ctx, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
