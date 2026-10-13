from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 40
    volatility_window: int = 20
    persistence_window: int = 10
    strength_threshold: float = 1.5
    persistence_threshold: float = 0.8
    profit_target: float = 0.05
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791331220"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.fractional_window + params.persistence_window - 1,
                       params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.volatility_window,
               params.persistence_window) < 2:
            raise ValueError("Indicator windows must be at least two")
        close = data["close"].astype(float).where(data["close"] > 0)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = weights[k - 1] * (k - 1 - params.fractional_order) / k
        fractional_close = close.copy()
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        # Remove the finite kernel's constant-price residue using its oldest observation.
        out["fractional_trend"] = fractional_close - weights.sum() * close.shift(
            params.fractional_window - 1)
        out["daily_change"] = close.diff()
        out["change_volatility"] = out["daily_change"].rolling(
            params.volatility_window).std(ddof=0)
        out["strength"] = out["fractional_trend"] / out["change_volatility"].replace(0, np.nan)
        out["direction"] = np.sign(out["fractional_trend"])
        out["persistence"] = out["direction"].rolling(params.persistence_window).mean()
        out["ma200"] = close.rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.max_hold_bars < 1 or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive")
        if params.strength_threshold <= 0 or not 0 < params.persistence_threshold <= 1:
            raise ValueError("Invalid entry thresholds")
        long_entry = ((data["close"] > indicators["ma200"])
                      & (indicators["strength"] >= params.strength_threshold)
                      & (indicators["persistence"] >= params.persistence_threshold))
        short_entry = ((data["close"] < indicators["ma200"])
                       & (indicators["strength"] <= -params.strength_threshold)
                       & (indicators["persistence"] <= -params.persistence_threshold))
        desired = np.select([long_entry.fillna(False), short_entry.fillna(False)],
                            [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                # The preceding close's decision fills at this bar's open.
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_bar, entry_price = i, opens[i]
                held_bars = i - entry_bar + 1
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held_bars >= params.max_hold_bars:
                    position, entry_bar = 0, -1
                    # Preserve a flat bar before permitting another entry.
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = int(desired[i])
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=8)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for side in (1, -1):
        price = 100.0 + side
        data = pd.DataFrame({"open": price, "high": price, "low": price,
                             "close": price, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"ma200": 100.0, "strength": 2.0 * side,
                            "persistence": float(side)}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist()[:5] == [0, side, side, side, 0]
        # A different fill price must govern the target, not the decision close.
        data.loc[index[1], "open"] = price * (1.0 - side * 0.10)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist()[:3] == [0, side, 0]
        assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=260))
    ind = strategy.indicators(flat, params)
    assert not strategy.generate_signals(flat, ind, ctx, params).data["signal"].any()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
