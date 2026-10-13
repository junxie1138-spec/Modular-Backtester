from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    min_roofs: int = 3
    profit_target: float = 0.03
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791287779"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        x = np.column_stack([close.shift(7 - k).to_numpy(dtype=float) for k in range(8)])
        valid = np.isfinite(x).all(axis=1)
        roofs = np.zeros(len(data), dtype=int)
        # Horizontal visibility: every intervening close is below both endpoints.
        # Count edges crossing the candidate trough, two sessions before today.
        for left in range(5):
            for right in (6, 7):
                roofs += x[:, left + 1:right].max(axis=1) < np.minimum(x[:, left], x[:, right])
        recovery = valid & (x[:, 5] == x.min(axis=1)) & (x[:, 6] > x[:, 5]) & (x[:, 7] > x[:, 6])
        mean = x.mean(axis=1)
        discount = np.divide(mean - x[:, 7], mean, out=np.full(len(data), np.nan), where=valid & (mean > 0))
        return pd.DataFrame({
            "roof_count": np.where(valid, roofs, np.nan),
            "trough_recovery": recovery,
            "discount": discount,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        if not 1 <= params.min_roofs <= 10:
            raise ValueError("min_roofs must be between 1 and 10")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not 15 <= params.max_hold <= 20:
            raise ValueError("max_hold must be between 15 and 20")
        entry = (
            indicators["roof_count"].ge(params.min_roofs)
            & indicators["trough_recovery"].fillna(False)
            & indicators["discount"].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar == -1:
                    # Yesterday's decision fills at today's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError("Entry open must be finite and positive")
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    active = False
                    entry_bar = -1
                    # Exit decided at this close, filled at the next open.
                    continue
            elif entry[i]:
                active = True
                entry_bar = -1
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TopologyParams()
    for target_test in (False, True):
        index = pd.date_range("2020-01-01", periods=24)
        data = pd.DataFrame({name: 100.0 for name in ("open", "high", "low", "close", "volume")}, index=index)
        if target_test:
            data.loc[index[2], ["high", "close"]] = 104.0
        ind = pd.DataFrame({"roof_count": 0.0, "trough_recovery": False, "discount": 0.0}, index=index)
        ind.loc[index[0], ["roof_count", "trough_recovery", "discount"]] = [3.0, True, 0.01]
        result = strategy.generate_signals(data, ind, None, params).data
        exit_bar = 3 if target_test else params.max_hold + 1
        assert result["signal"].iloc[0] == 0
        assert result["signal"].iloc[1:exit_bar].eq(1).all()
        assert result["signal"].iloc[exit_bar:].eq(0).all()
        assert result["size"].gt(0).all()
    strategy.indicators(data, params)
