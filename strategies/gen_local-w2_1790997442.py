from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    min_degree: int = 5
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1790997442"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return 17

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = close.shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["three_day_change"] = close - close.shift(3)
        out["body"] = close - data["open"]

        # Horizontal visibility on inverted closes: intervening prices
        # must exceed both endpoint prices. Only backward edges are used.
        interior_min = pd.Series(np.inf, index=data.index, dtype=float)
        degree = pd.Series(0, index=data.index, dtype=int)
        valid = pd.Series(np.isfinite(close), index=data.index)
        for lag in range(1, 17):
            past = close.shift(lag)
            valid &= np.isfinite(past)
            visible = (np.maximum(close, past) < interior_min)
            degree += visible.fillna(False).astype(int)
            interior_min = pd.Series(np.minimum(interior_min, past), index=data.index)
        out["downside_degree"] = degree.astype(float).where(valid)
        out["entry_candidate"] = (
            out["downside_degree"].ge(params.min_degree)
            & out["three_day_change"].lt(0)
            & out["body"].gt(0)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["entry_candidate"].to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_bar = -1
        fixed_stop = np.nan

        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open. Freeze
            # the stop using that fill and yesterday's available ATR.
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]

            if active:
                stop_hit = np.isfinite(close[i]) and close[i] < fixed_stop
                holding_expired = i - entry_bar + 1 >= 4
                if stop_hit or holding_expired:
                    active = False
                else:
                    targets[i] = 1
            elif candidate[i] and i + 1 < len(data):
                targets[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
