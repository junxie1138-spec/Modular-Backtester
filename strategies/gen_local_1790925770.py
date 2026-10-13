from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    acceleration_floor: float = 0.2
    contraction_ceiling: float = 0.9
    breakeven_pct: float = 0.008
    atr_multiple: float = 1.7
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790925770"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.roc_bars + 2, 18))

    def indicators(self, data, params):
        if params.roc_bars < 1 or not 3 <= params.holding_bars <= 5:
            raise ValueError("roc_bars must be positive; holding_bars must be 3 through 5")
        if not all(np.isfinite(x) and x > 0 for x in (
            params.acceleration_floor, params.contraction_ceiling,
            params.breakeven_pct, params.atr_multiple
        )):
            raise ValueError("Thresholds must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["roc_acceleration"] = out["roc"].diff()
        out["acceleration"] = out["roc_acceleration"] / (
            out["atr"] / close.replace(0, np.nan)
        ).replace(0, np.nan)
        out["range_fraction"] = out["true_range"] / out["previous_close"].replace(0, np.nan)
        out["recent_range"] = out["range_fraction"].rolling(3, min_periods=3).mean()
        out["baseline_range"] = out["range_fraction"].rolling(14, min_periods=14).mean().shift(3)
        out["contraction"] = out["recent_range"] / out["baseline_range"].replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = np.isfinite(indicators).all(axis=1)
        squeeze = ready & (indicators["contraction"] < params.contraction_ceiling)
        long = squeeze & (data["close"] > indicators["ma200"]) & (
            indicators["roc"] > 0
        ) & (indicators["acceleration"] > params.acceleration_floor)
        short = squeeze & (data["close"] < indicators["ma200"]) & (
            indicators["roc"] < 0
        ) & (indicators["acceleration"] < -params.acceleration_floor)
        candidates = pd.Series(np.select([long, short], [1, -1], default=0), index=data.index)
        pulses = candidates.where(candidates != candidates.shift(1).fillna(0), 0).to_numpy()
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        entry, stop, extreme = np.nan, np.nan, np.nan
        armed = False
        for i, (opening, high, low, close) in enumerate(prices):
            if side:
                if entry_bar == -1:
                    if not np.isfinite(opening) or opening <= 0:
                        side = 0
                        continue
                    entry, extreme, entry_bar = opening, opening, i
                    stop = entry - side * params.atr_multiple * atr[i - 1]
                    armed = False
                # Stops are close-time decisions; liquidation fills at the next open.
                hit = (low <= stop) if side == 1 else (high >= stop)
                if np.isfinite(high) and np.isfinite(low) and np.isfinite(close):
                    extreme = max(extreme, high) if side == 1 else min(extreme, low)
                    armed = armed or side * (extreme / entry - 1) >= params.breakeven_pct
                    if armed:
                        proposed = entry
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            trail = extreme - side * params.atr_multiple * atr[i]
                            proposed = max(entry, trail) if side == 1 else min(entry, trail)
                        # Short stops ratchet downward, the favorable mirror of long stops.
                        stop = max(stop, proposed) if side == 1 else min(stop, proposed)
                        hit = hit or side * (close - stop) <= 0
                if hit or i - entry_bar + 1 >= params.holding_bars:
                    side, entry_bar = 0, -1
                    continue
            elif pulses[i] and np.isfinite(prices[i]).all() and atr[i] > 0:
                side, entry_bar = int(pulses[i]), -1
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=6)
    base = pd.DataFrame({
        "open": 100.0,
        "high": [100.5, 102.5, 103.5, 100.5, 100.5, 100.5],
        "low": [99.5, 101.5, 102.5, 99.5, 99.5, 99.5],
        "close": [100.0, 102.0, 103.0, 100.0, 100.0, 100.0],
        "volume": 1000.0
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for side in (1, -1):
        data = base.copy()
        if side == -1:
            data["open"] = 200 - base["open"]
            data["high"] = 200 - base["low"]
            data["low"] = 200 - base["high"]
            data["close"] = 200 - base["close"]
        ind = pd.DataFrame({
            "atr": 1.0, "ma200": 100.0 - side, "roc": side * 0.01,
            "acceleration": float(side), "contraction": [0.5, 2, 2, 2, 2, 2]
        }, index=index)
        signals = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signals.tolist() == [0, side, side, side, 0, 0]
        prefix = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], ctx, params).data
        assert prefix["signal"].tolist() == signals.iloc[:3].tolist()
        warm = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params)
        assert warm.data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
