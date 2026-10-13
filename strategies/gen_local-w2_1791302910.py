from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 24
    atr_window: int = 14
    bridge_threshold: float = 0.15
    edge_fraction: float = 0.40
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791302910"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.graph_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.graph_window >= 4 and params.atr_window >= 1
                and 0 < params.bridge_threshold < 1
                and 0 < params.edge_fraction < 0.5
                and params.breakeven_pct > 0 and params.trail_atr > 0
                and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        tr = tr.where(previous.notna())
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        bottom = data["low"].rolling(params.graph_window).min()
        top = data["high"].rolling(params.graph_window).max()
        out["relative_position"] = (close - bottom) / (top - bottom).replace(0, np.nan)
        upper_span = pd.Series(0.0, index=data.index)
        lower_span = upper_span.copy()
        maximum = pd.Series(-np.inf, index=data.index)
        minimum = pd.Series(np.inf, index=data.index)
        complete = close.notna()
        # Natural visibility: each edge clears every intervening slope.
        # ponytail: O(bars * graph_window); keep windows small, use a compiled sweep for large windows.
        for distance in range(1, params.graph_window):
            slope = (close.shift(distance) - close) / distance
            complete &= slope.notna()
            if distance >= params.graph_window // 2:
                upper_span += (slope > maximum).astype(float) * distance
                lower_span += (slope < minimum).astype(float) * distance
            maximum = pd.concat([maximum, slope], axis=1).max(axis=1)
            minimum = pd.concat([minimum, slope], axis=1).min(axis=1)
        total = upper_span + lower_span
        out["upper_span"] = upper_span.where(complete)
        out["lower_span"] = lower_span.where(complete)
        out["bridge_balance"] = ((upper_span - lower_span) / total.replace(0, np.nan)).where(complete)
        out["close_change"] = close.diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        balance = indicators["bridge_balance"]
        ready = indicators[["ma200", "atr", "relative_position", "bridge_balance",
                            "close_change"]].notna().all(axis=1) & (indicators["atr"] > 0)
        long_entry = (ready & (data["close"] > indicators["ma200"])
                      & (indicators["relative_position"] <= params.edge_fraction)
                      & (indicators["close_change"] > 0)
                      & (balance > params.bridge_threshold)
                      & (balance.shift(1) <= params.bridge_threshold))
        short_entry = (ready & (data["close"] < indicators["ma200"])
                       & (indicators["relative_position"] >= 1 - params.edge_fraction)
                       & (indicators["close_change"] < 0)
                       & (balance < -params.bridge_threshold)
                       & (balance.shift(1) >= -params.bridge_threshold))
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, stop, best = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if side:
                if not (np.isfinite(prices[i]).all() and np.isfinite(atr[i])
                        and np.isfinite(ma[i]) and atr[i] > 0):
                    side = 0
                    continue
                if np.isnan(entry):
                    # The preceding close's instruction fills at this bar's open.
                    entry = opening
                    best = side * entry
                    stop = best - params.trail_atr * atr[i - 1]
                age += 1
                adverse = low if side == 1 else -high
                regime_lost = side * (close - ma[i]) <= 0
                # Check the previously established stop before updating it.
                if adverse <= stop or age >= params.max_hold or regime_lost:
                    side = 0
                else:
                    best = max(best, high if side == 1 else -low)
                    armed = armed or best - side * entry >= entry * params.breakeven_pct
                    if armed:
                        # Signed-price stops only rise; a short's actual price stop only falls.
                        stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                signals[i] = side
                continue
            if candidates[i] and np.isfinite(prices[i]).all():
                side = int(candidates[i])
                age, armed = 0, False
                entry, stop, best = np.nan, np.nan, np.nan
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                             "close": 100.0, "volume": 1000.0}, index=index)
        data.loc[index[2], "high" if direction == 1 else "low"] = 102.0 if direction == 1 else 98.0
        derived = pd.DataFrame({"ma200": 90.0 if direction == 1 else 110.0,
                                "atr": 1.0, "relative_position": 0.2 if direction == 1 else 0.8,
                                "bridge_balance": [0.0, 0.5 * direction, 0.5 * direction,
                                                   0.5 * direction, 0.5 * direction],
                                "close_change": float(direction)}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0]
        assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
