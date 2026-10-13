from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRelaxationParams:
    slow_window: int = 24
    fast_window: int = 4
    atr_window: int = 14
    gap_dominance: float = 0.9
    collapse_ratio: float = 0.6
    holding_bars: int = 18
    breakeven_pct: float = 0.03
    initial_atr: float = 3.0
    trail_atr: float = 2.5
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[GapRelaxationParams]):
    strategy_id = "gen_local_1790885981"

    @classmethod
    def params_type(cls):
        return GapRelaxationParams

    def warmup_bars(self, params: GapRelaxationParams) -> int:
        return int(max(params.slow_window, params.fast_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapRelaxationParams) -> pd.DataFrame:
        if not (2 <= params.fast_window <= params.slow_window and params.atr_window >= 2):
            raise ValueError("Require 2 <= fast_window <= slow_window and atr_window >= 2")
        if not (0 < params.collapse_ratio < 1 and params.gap_dominance > 0):
            raise ValueError("Invalid volatility thresholds")
        if not (params.holding_bars >= 1 and params.breakeven_pct > 0
                and params.initial_atr > 0 and params.trail_atr > 0
                and params.entry_slippage_bps >= 0):
            raise ValueError("Invalid holding or exit parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap_return"] = (data["open"] / out["previous_close"].where(out["previous_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["day_return"] = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_fast_std"] = out["gap_return"].rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["gap_slow_std"] = out["gap_return"].rolling(params.slow_window, min_periods=params.slow_window).std(ddof=0)
        out["day_fast_std"] = out["day_return"].rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["day_slow_std"] = out["day_return"].rolling(params.slow_window, min_periods=params.slow_window).std(ddof=0)
        out["gap_relaxation"] = out["gap_fast_std"] / out["gap_slow_std"].where(out["gap_slow_std"] > 0)
        out["overnight_dominance"] = out["gap_slow_std"] / out["day_slow_std"].where(out["day_slow_std"] > 0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["cluster_mean"] = data["close"].rolling(params.slow_window, min_periods=params.slow_window).mean()
        out["qualified"] = (
            (out["gap_relaxation"] < params.collapse_ratio)
            & (out["overnight_dominance"] >= params.gap_dominance)
            & (out["day_fast_std"] <= out["day_slow_std"])
            & (data["close"] > out["cluster_mean"])
            & (out["day_return"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        out["trigger"] = out["qualified"] & ~out["qualified"].shift(1, fill_value=False)
        out["conviction"] = (
            0.25 + 0.75 * (1 - out["gap_relaxation"]).clip(0, 1)
            * (out["overnight_dominance"] / params.gap_dominance).clip(1, 2) / 2
        ).replace([np.inf, -np.inf], np.nan).fillna(0.25).clip(0.25, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRelaxationParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        triggers = indicators["trigger"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        active = pending = armed = False
        entry = peak = stop = 0.0
        held_size = 1.0
        age = 0
        for i in range(n):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # Entry uses this bar's open, never a future open at decision time.
                    entry = opens[i] * (1 + params.entry_slippage_bps / 10000)
                    peak = entry
                    stop = entry - params.initial_atr * atr[i - 1]
                    age = 0
                    active = True
                    armed = False
            if active:
                age += 1
                sizes[i] = held_size
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    active = False
                    continue
                peak = max(peak, closes[i])
                if peak >= entry * (1 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Stops are evaluated on closes; the zero target is executed next bar.
                if closes[i] <= stop or age >= params.holding_bars:
                    active = False
                else:
                    signals[i] = 1
                continue
            if triggers[i] and np.isfinite(conviction[i]):
                held_size = float(np.clip(conviction[i], 0.25, 1.0))
                signals[i] = 1
                sizes[i] = held_size
                pending = True
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRelaxationParams()
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    close = np.array([100, 101, 104, 108, 106, 105, 104, 103], dtype=float)
    data = pd.DataFrame({"open": 100.0, "high": close + 1, "low": 99.0,
                         "close": close, "volume": 1000.0}, index=idx)
    ind = pd.DataFrame({"atr": [1, 1, 1, 1, 5, 5, 5, 5],
                        "trigger": [True, False, False, False, False, False, False, False],
                        "conviction": 0.4}, index=idx)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    assert result["size"].iloc[1] == 0.4
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], ind.iloc[:end], ctx, params).data
        assert prefix.equals(result.iloc[:end])
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=80, freq="B"))
    flat_ind = strategy.indicators(flat, params)
    assert not flat_ind["trigger"].any()
    assert not strategy.generate_signals(flat, flat_ind, ctx, params).data["signal"].any()
    assert strategy.warmup_bars(params) >= max(params.slow_window, params.atr_window) + 1
    timed = GapRelaxationParams(holding_bars=3)
    assert strategy.generate_signals(data, ind, ctx, timed).data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
