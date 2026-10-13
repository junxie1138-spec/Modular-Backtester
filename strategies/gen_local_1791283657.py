from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 8
    acceleration_floor: float = 0.003
    atr_bars: int = 14
    breakeven_gain: float = 0.02
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791283657"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(2 * params.roc_bars + 2, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.roc_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(x) for x in
                   (params.acceleration_floor, params.breakeven_gain, params.trail_atr)):
            raise ValueError("Thresholds must be finite.")
        if params.acceleration_floor < 0 or params.breakeven_gain <= 0 or params.trail_atr <= 0:
            raise ValueError("Acceleration must be nonnegative; stop parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["daily_return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["previous_roc"] = out["roc"].shift(params.roc_bars)
        out["best_gain"] = out["daily_return"].rolling(
            params.roc_bars, min_periods=params.roc_bars
        ).max().clip(lower=0.0)
        out["trimmed_roc"] = (1.0 + out["roc"]) / (1.0 + out["best_gain"]) - 1.0
        out["acceleration"] = out["roc"] - out["previous_roc"]
        out["robust_acceleration"] = out["trimmed_roc"] - out["previous_roc"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["qualified"] = (
            (out["roc"] > 0)
            & (out["acceleration"] > params.acceleration_floor)
            & (out["robust_acceleration"] > params.acceleration_floor)
            & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["qualified"] & out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry = stop = peak = entry_atr = 0.0
        age = 0
        protected = False
        for i in range(len(data)):
            if pending:
                # Yesterday's decision fills at today's open; never read a future row.
                entry = opening[i]
                pending = False
                if not np.isfinite(entry) or entry <= 0:
                    active = False
                else:
                    peak = entry
                    stop = max(0.0, entry - params.trail_atr * entry_atr)
                    age = 0
                    protected = False
            if active:
                age += 1
                if not np.isfinite(close[i]) or close[i] <= 0:
                    active = False
                else:
                    peak = max(peak, close[i])
                    if close[i] >= entry * (1.0 + params.breakeven_gain):
                        protected = True
                    if protected:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                    # Close-observed stops become next-open market exits.
                    if close[i] <= stop or age >= params.hold_bars:
                        active = False
                decisions[i] = int(active)
                continue
            if confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                if np.isfinite(close[i]) and close[i] > 0:
                    active = pending = True
                    entry_atr = atr[i]
                    decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    prices = np.array([100, 100, 100, 105, 106, 103.5, 102.5, 102], dtype=float)
    data = pd.DataFrame({
        "open": [100, 100, 100, 102, 105, 106, 103.5, 102.5],
        "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1000.0,
    }, index=index)
    flags = pd.Series([False, True, True, False, False, False, False, False], index=index)
    derived = pd.DataFrame({
        "atr": [1, 1, 1, 1, 1, 10, 10, 10],
        "confirmed": flags & flags.shift(1, fill_value=False),
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, derived, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert (signals["size"] > 0).all()
    params.hold_bars = 3
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].tolist() == [
        0, 0, 0, 1, 1, 1, 0, 0
    ]
    actual = strategy.indicators(data, StrategyParams(roc_bars=2, atr_bars=2))
    assert not actual["confirmed"].iloc[:5].any()
    assert actual.index.equals(data.index)


if __name__ == "__main__":
    _self_check()
