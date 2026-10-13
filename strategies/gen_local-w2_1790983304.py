from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 63
    repair_days: int = 4
    min_depth: float = 0.025
    max_depth: float = 0.12
    remaining_fraction: float = 0.65
    participation_window: int = 8
    min_effective_days: float = 5.0
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 18

    def __post_init__(self):
        if min(self.peak_window, self.repair_days, self.participation_window,
               self.atr_window, self.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if not 0 < self.min_depth < self.max_depth < 1:
            raise ValueError("Invalid drawdown bounds.")
        if not 0 < self.remaining_fraction < 1:
            raise ValueError("Recovery fraction must be between zero and one.")
        if not 0 < self.min_effective_days <= self.participation_window:
            raise ValueError("Effective days must fit the participation window.")
        if not self.breakeven_pct > 0 or not self.trail_atr > 0:
            raise ValueError("Exit parameters must be positive.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790983304"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.peak_window + params.repair_days,
                       params.participation_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["reference_peak"] = close.shift(params.repair_days).rolling(
            params.peak_window, min_periods=params.peak_window).max()
        out["initial_depth"] = 1.0 - close.shift(params.repair_days) / out["reference_peak"]
        out["current_depth"] = 1.0 - close / out["reference_peak"]
        # Effective count rewards advances spread over many volume-bearing sessions.
        change = close.diff()
        volume = data["volume"].clip(lower=0)
        out["advance_volume"] = volume.where(change > 0, 0.0).where(change.notna())
        out["advance_volume_sum"] = out["advance_volume"].rolling(
            params.participation_window, min_periods=params.participation_window).sum()
        out["advance_volume_square_sum"] = out["advance_volume"].pow(2).rolling(
            params.participation_window, min_periods=params.participation_window).sum()
        out["effective_days"] = out["advance_volume_sum"].pow(2) / out[
            "advance_volume_square_sum"].replace(0.0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).where(close.shift(1).notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["depth_agrees"] = (
            out["initial_depth"].between(params.min_depth, params.max_depth)
            & (out["current_depth"] >= 0)
            & (out["current_depth"] <= out["initial_depth"] * params.remaining_fraction)
        )
        out["participation_agrees"] = out["effective_days"] >= params.min_effective_days
        out["entry"] = (out["depth_agrees"] & out["participation_agrees"]
                        & out["atr"].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry_price = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # The preceding close's decision fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    stop = entry_price - params.trail_atr * atrs[i - 1]
            if active:
                price = closes[i]
                if (not np.isfinite(price) or price <= 0 or price <= stop
                        or i - entry_bar + 1 >= params.max_hold):
                    active = False
                    continue
                if price >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, price - params.trail_atr * atrs[i])
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 98.0,
                         "close": [99.0, 100.0, 104.0, 105.0, 101.0, 100.0, 100.0],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
                               "entry": [True, False, False, False, False, False, False]},
                              index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert (result["size"] > 0).all()
