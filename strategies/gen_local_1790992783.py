from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBalanceParams:
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapBalanceParams]):
    strategy_id = "gen_local_1790992783"

    @classmethod
    def params_type(cls):
        return GapBalanceParams

    def warmup_bars(self, params: GapBalanceParams) -> int:
        return 14

    def indicators(self, data: pd.DataFrame, params: GapBalanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(np.isfinite(data["open"]) & (data["open"] > 0))
        closing = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["prior_close"] = closing.shift(1)
        out["gap"] = (opening / out["prior_close"] - 1.0).replace([np.inf, -np.inf], np.nan)
        out["daylight"] = (closing / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        out["absolute_gap"] = out["gap"].abs()
        out["positive_gap"] = out["gap"].gt(0).astype(float).where(out["gap"].notna())
        out["negative_gap"] = out["gap"].lt(0).astype(float).where(out["gap"].notna())
        out["baseline_net"] = out["gap"].rolling(10, min_periods=10).sum().shift(3)
        out["baseline_gross"] = out["absolute_gap"].rolling(10, min_periods=10).sum().shift(3)
        out["baseline_balance"] = out["baseline_net"].abs() / out["baseline_gross"].replace(0.0, np.nan)
        out["baseline_mean_gap"] = out["baseline_gross"] / 10.0
        out["baseline_positive_count"] = out["positive_gap"].rolling(10, min_periods=10).sum().shift(3)
        out["baseline_negative_count"] = out["negative_gap"].rolling(10, min_periods=10).sum().shift(3)
        out["recent_gap_sum"] = out["gap"].rolling(3, min_periods=3).sum()
        out["recent_positive_count"] = out["positive_gap"].rolling(3, min_periods=3).sum()
        out["recent_daylight_sum"] = out["daylight"].rolling(3, min_periods=3).sum()
        out["entry"] = (
            out["baseline_balance"].le(0.15)
            & out["baseline_positive_count"].ge(3)
            & out["baseline_negative_count"].ge(3)
            & out["recent_positive_count"].ge(2)
            & out["recent_gap_sum"].gt(3.0 * out["baseline_mean_gap"])
            & out["recent_daylight_sum"].le(0.0)
            & out["recent_daylight_sum"].gt(-0.5 * out["recent_gap_sum"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBalanceParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        entry_price = None
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # Observe the entry open only when its bar arrives.
                entry_price = opens[i]
                age = 0
            if entry_price is not None:
                age += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.hold_bars:
                    entry_price = None
                    age = 0
                else:
                    raw[i] = 1
            elif entries[i] and i >= self.warmup_bars(params) - 1:
                raw[i] = 1
                pending = True
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
