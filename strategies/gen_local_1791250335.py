from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fractional_order: float = 0.4
    fractional_window: int = 40
    efficiency_window: int = 12
    entry_efficiency: float = 0.35
    volatility_window: int = 20
    target_volatility: float = 0.12
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791250335"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.fractional_window + params.efficiency_window,
                       params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (0 < params.fractional_order < 1 and params.fractional_window >= 2
                and params.efficiency_window >= 2 and params.volatility_window >= 2
                and 0 < params.entry_efficiency <= 1 and params.target_volatility > 0
                and params.profit_target > 0 and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        # ponytail: fixed finite fractional memory; enlarge the window for longer memory.
        weights = np.ones(params.fractional_window)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(k)
        out["fractional_close"] = fractional
        out["fractional_step"] = fractional.diff()
        out["net_motion"] = out["fractional_step"].rolling(params.efficiency_window).sum()
        out["path_length"] = out["fractional_step"].abs().rolling(params.efficiency_window).sum()
        out["efficiency"] = out["net_motion"] / out["path_length"].replace(0.0, np.nan)
        out["ma200"] = close.rolling(200).mean()
        out["returns"] = close.pct_change(fill_method=None)
        out["annual_volatility"] = out["returns"].rolling(params.volatility_window).std() * np.sqrt(252.0)
        out["target_size"] = (params.target_volatility / out["annual_volatility"].replace(0.0, np.nan)).clip(0.01, 0.95)
        ready = np.isfinite(out["efficiency"]) & np.isfinite(out["target_size"]) & np.isfinite(out["ma200"])
        out["entry"] = np.select(
            [ready & (close > out["ma200"]) & (out["efficiency"] >= params.entry_efficiency),
             ready & (close < out["ma200"]) & (out["efficiency"] <= -params.entry_efficiency)],
            [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.01, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        risk_sizes = indicators["target_size"].to_numpy(dtype=float)
        side = 0
        decision_bar = -1
        entry_price = np.nan
        held_size = 0.01
        for i in range(n):
            if side:
                # Observe the opening fill only after its bar has arrived.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or i - decision_bar >= params.max_hold:
                    side = 0
                # Exits have priority: no same-close reentry.
            elif entries[i] and np.isfinite(risk_sizes[i]) and risk_sizes[i] > 0:
                side = int(entries[i])
                decision_bar = i
                entry_price = np.nan
                # Freeze volatility-targeted exposure for this holding period.
                held_size = float(np.clip(risk_sizes[i], 0.01, 0.95))
            signals[i] = side
            sizes[i] = held_size if side else 0.01
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=2)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({name: np.full(8, 100.0) for name in
                         ("open", "high", "low", "close", "volume")}, index=index)
    ind = pd.DataFrame({"entry": [1, 0, 0, 0, -1, 0, 0, 0],
                        "target_size": 0.5}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result.loc[index[1], "size"] == 0.5
    data.loc[index[1], "close"] = 105.0
    data.loc[index[5], "close"] = 95.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, 0, 0]
    flat = pd.DataFrame({name: np.full(260, 100.0) for name in
                         ("open", "high", "low", "close", "volume")},
                        index=pd.date_range("2020-01-01", periods=260))
    computed = strategy.indicators(flat, params)
    assert not computed["entry"].any()
    assert not strategy.generate_signals(flat, computed, ctx, params).data["signal"].any()
