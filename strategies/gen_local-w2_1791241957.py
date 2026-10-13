from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.45
    frac_taps: int = 48
    baseline_bars: int = 20
    box_bars: int = 6
    compression_atr: float = 3.0
    excursion_z: float = 1.0
    atr_bars: int = 20
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791241957"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.frac_taps + params.baseline_bars,
                       params.atr_bars + 1, params.box_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_taps, params.baseline_bars, params.box_bars,
               params.atr_bars, params.max_hold) < 2:
            raise ValueError("Lookbacks and max_hold must be at least two")
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.compression_atr, params.excursion_z, params.trail_atr)):
            raise ValueError("Thresholds must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        # Fixed finite history makes fractional differentiation causal and reproducible.
        weights = np.ones(params.frac_taps, dtype=float)
        for lag in range(1, params.frac_taps):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        fractional = close * weights[0]
        for lag in range(1, params.frac_taps):
            fractional = fractional + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional
        history = fractional.shift(1).rolling(params.baseline_bars)
        out["fd_mean"] = history.mean()
        out["fd_std"] = history.std(ddof=0)
        out["fd_z"] = ((fractional - out["fd_mean"]) /
                       out["fd_std"].replace(0.0, np.nan))
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["box_high"] = data["high"].shift(1).rolling(params.box_bars).max()
        out["box_low"] = data["low"].shift(1).rolling(params.box_bars).min()
        out["box_width"] = out["box_high"] - out["box_low"]
        out["compression"] = out["box_width"] / out["atr"].shift(1).replace(0.0, np.nan)
        valid = np.isfinite(out[["fd_z", "atr", "compression", "box_high", "box_low"]]).all(axis=1)
        compressed = valid & (out["atr"] > 0) & (out["compression"] <= params.compression_atr)
        out["entry_long"] = compressed & (out["fd_z"] >= params.excursion_z) & (close > out["box_high"])
        out["entry_short"] = compressed & (out["fd_z"] <= -params.excursion_z) & (close < out["box_low"])
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["entry_long"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["entry_short"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position, held = 0, 0
        extreme, stop = np.nan, np.nan
        for i in range(len(data)):
            usable = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                held += 1
                if not usable:
                    position = 0
                else:
                    # Start the watermark on the fill bar, never on the decision bar.
                    if position == 1:
                        extreme = close[i] if np.isnan(extreme) else max(extreme, close[i])
                        candidate = extreme - params.trail_atr * atr[i]
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        stopped = close[i] <= stop
                    else:
                        extreme = close[i] if np.isnan(extreme) else min(extreme, close[i])
                        candidate = extreme + params.trail_atr * atr[i]
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        stopped = close[i] >= stop
                    if stopped or held >= params.max_hold:
                        position = 0
                # An exit always emits flat; no reversal or re-entry on this decision bar.
            elif usable:
                if longs[i] != shorts[i]:
                    position = 1 if longs[i] else -1
                    held = 0
                    extreme, stop = np.nan, np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100., 101., 104., 102., 101., 98., 101., 101.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 3., 1., 1., 3., 1.],
        "entry_long": [True, False, False, False, False, False, False, False],
        "entry_short": [False, False, False, False, True, False, False, False]
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    params.max_hold = 2
    indicators["entry_short"] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
