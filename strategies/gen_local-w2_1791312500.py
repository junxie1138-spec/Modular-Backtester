from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCatchupParams:
    lookback: int = 3
    atr_period: int = 14
    min_momentum: float = 0.004
    min_efficiency: float = 0.7
    gap_burden: float = 0.004
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    initial_stop_atr: float = 1.8
    max_hold: int = 4

    def __post_init__(self):
        if self.lookback < 2 or self.atr_period < 2 or not 3 <= self.max_hold <= 5:
            raise ValueError("Invalid lookback or holding horizon")
        values = (self.min_momentum, self.gap_burden, self.breakeven_pct,
                  self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("Thresholds must be positive and finite")
        if not 0 < self.min_efficiency <= 1:
            raise ValueError("Efficiency must be in (0, 1]")


class GeneratedStrategy(BaseStrategy[GapCatchupParams]):
    strategy_id = "gen_local-w2_1791312500"

    @classmethod
    def params_type(cls):
        return GapCatchupParams

    def warmup_bars(self, params: GapCatchupParams) -> int:
        return int(max(200, params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapCatchupParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1).replace(0, np.nan)
        returns = close / previous - 1.0
        gaps = data["open"] / previous - 1.0
        momentum = returns.shift(1).rolling(params.lookback, min_periods=params.lookback).sum()
        travel = returns.abs().shift(1).rolling(params.lookback, min_periods=params.lookback).sum()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        return pd.DataFrame({
            "return": returns,
            "gap": gaps,
            "momentum": momentum,
            "efficiency": momentum.abs() / travel.replace(0, np.nan),
            "gap_debt": gaps.shift(1).rolling(params.lookback, min_periods=params.lookback).sum(),
            "atr": true_range.rolling(params.atr_period, min_periods=params.atr_period).mean(),
            "ma200": close.rolling(200, min_periods=200).mean(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapCatchupParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0)
        coherent = indicators["efficiency"] >= params.min_efficiency
        long_entry = (
            ready & coherent & (data["close"] > indicators["ma200"])
            & (indicators["momentum"] >= params.min_momentum)
            & (indicators["gap_debt"] <= -params.gap_burden)
            & (indicators["gap"] > 0) & (indicators["return"] > 0)
        )
        short_entry = (
            ready & coherent & (data["close"] < indicators["ma200"])
            & (indicators["momentum"] <= -params.min_momentum)
            & (indicators["gap_debt"] >= params.gap_burden)
            & (indicators["gap"] < 0) & (indicators["return"] < 0)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        valid = ready.to_numpy()
        targets = np.zeros(len(data), dtype=int)
        direction, age = 0, 0
        entry, peak, stop = 0.0, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            if direction:
                if age == 0:
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        direction = 0
                        continue
                    peak = direction * entry
                    stop = peak - params.initial_stop_atr * atrs[i - 1]
                    armed = False
                age += 1
                if not valid[i] or not np.isfinite(closes[i]) or closes[i] <= 0:
                    direction = 0
                    continue
                price = direction * closes[i]
                peak = max(peak, price)
                armed = armed or direction * (closes[i] / entry - 1.0) >= params.breakeven_pct
                if armed:
                    # Favorable-price coordinates ratchet upward for both sides;
                    # the corresponding short buy-stop only moves downward.
                    stop = max(stop, direction * entry, peak - params.trail_atr * atrs[i])
                # Stops use closing prices; exits fill at the next open.
                if price <= stop or age >= params.max_hold:
                    direction = 0
                    continue
                targets[i] = direction
            elif candidates[i]:
                direction = int(candidates[i])
                age = 0
                targets[i] = direction
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapCatchupParams()
    for direction in (1, -1):
        close = 100.0 * np.exp(direction * 0.003 * np.arange(220))
        previous = np.r_[close[0], close[:-1]]
        opening = previous * (1.0 - direction * 0.002)
        opening[203] = previous[203] * (1.0 + direction * 0.001)
        data = pd.DataFrame({
            "open": opening,
            "high": np.maximum(opening, close) * 1.001,
            "low": np.minimum(opening, close) * 0.999,
            "close": close,
            "volume": 1000000.0,
        }, index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
        signals = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
        assert signals["signal"].iloc[203] == 0
        assert signals["signal"].iloc[204] == direction
        assert signals["signal"].iloc[207] == direction
        assert signals["signal"].iloc[208] == 0
        assert signals["signal"].isin([-1, 0, 1]).all() and (signals["size"] > 0).all()
        data.loc[data.index[206], "close"] = opening[204] * (1.0 - direction * 0.05)
        data["high"] = data[["open", "close"]].max(axis=1) * 1.001
        data["low"] = data[["open", "close"]].min(axis=1) * 0.999
        stopped = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
        assert stopped["signal"].iloc[207] == 0


if __name__ == "__main__":
    _self_check()
