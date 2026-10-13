from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_bars: int = 8
    atr_bars: int = 14
    min_gap_atr: float = 0.15
    min_body_fraction: float = 0.50
    breakeven_pct: float = 0.005
    trail_atr: float = 1.25
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791237880"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.range_bars, params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.range_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")
        if not (0 <= params.min_body_fraction <= 1):
            raise ValueError("Body fraction must be between zero and one.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.min_gap_atr, params.breakeven_pct, params.trail_atr)):
            raise ValueError("Gap, breakeven and trail parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars).min().shift(1)
        out["range_high"] = data["high"].rolling(params.range_bars).max().shift(1)
        out["midpoint"] = (out["range_low"] + out["range_high"]) / 2
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["body_fraction"] = ((data["close"] - data["open"]) /
                               (data["high"] - data["low"]).replace(0, np.nan))
        out["gap_agrees"] = (
            (out["gap"] <= -params.min_gap_atr * out["previous_atr"])
            & (out["previous_close"] >= out["midpoint"])
            & (data["open"] < out["midpoint"])
            & (data["open"] > out["range_low"])
        ).fillna(False)
        out["candle_agrees"] = (
            out["body_fraction"] >= params.min_body_fraction
        ).fillna(False)
        out["setup"] = (out["gap_agrees"] & out["candle_agrees"]
                        & (out["previous_atr"] > 0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            row = data.iloc[i]
            atr = float(indicators["atr"].iloc[i])
            if pending:
                # Today's open fills yesterday's shifted entry decision.
                entry = float(row["open"])
                initial_atr = float(indicators["atr"].iloc[i - 1])
                active = (np.isfinite(entry) and entry > 0
                          and np.isfinite(initial_atr) and initial_atr > 0)
                stop = entry - params.trail_atr * initial_atr
                peak, age, armed, pending = entry, 0, False, False
            if active:
                age += 1
                valid = (np.isfinite([row["open"], row["high"], row["low"],
                                     row["close"], atr]).all() and atr > 0)
                breached = not valid or float(row["low"]) <= stop
                if valid and not breached:
                    peak = max(peak, float(row["high"]))
                    armed = armed or peak >= entry * (1 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr)
                    # Test newly raised stops against the close: high/low order is unknown.
                    breached = float(row["close"]) <= stop
                if breached or age >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif bool(indicators["setup"].iloc[i]) and np.isfinite(atr) and atr > 0:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        # Stop breaches request an exit at the next open, not an intrabar fill.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=4)
    bars = pd.DataFrame({
        "open": [100.0, 100.0, 100.5, 100.0],
        "high": [100.2, 101.0, 100.6, 100.2],
        "low": [99.8, 99.8, 100.1, 99.8],
        "close": [100.0, 100.4, 100.4, 100.0],
        "volume": [1000.0] * 4,
    }, index=index)
    derived = pd.DataFrame({"atr": [0.4, 0.4, 2.0, 0.4],
                            "setup": [True, False, False, False]}, index=index)
    result = strategy.generate_signals(bars, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    expiry = strategy.generate_signals(
        bars, derived, None, Params(breakeven_pct=0.10)
    ).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(bars, Params()).index.equals(index)
