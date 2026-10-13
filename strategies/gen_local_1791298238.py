from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PostExpiryParams:
    frac_window: int = 20
    frac_order: float = 0.4
    profit_target: float = 0.018
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[PostExpiryParams]):
    strategy_id = "gen_local_1791298238"

    @classmethod
    def params_type(cls):
        return PostExpiryParams

    def warmup_bars(self, params: PostExpiryParams) -> int:
        return params.frac_window + 5

    def indicators(self, data: pd.DataFrame, params: PostExpiryParams) -> pd.DataFrame:
        weights = [1.0]
        for k in range(1, params.frac_window):
            weights.append(-weights[-1] * (params.frac_order - k + 1) / k)

        fd = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fd = fd + weight * data["close"].shift(k)

        weekday = data.index.dayofweek
        previous_friday = data.index.day - ((weekday - 4) % 7)
        out = pd.DataFrame(index=data.index)
        out["fd_close"] = fd
        out["post_expiry"] = (weekday <= 2) & (previous_friday >= 15) & (previous_friday <= 21)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PostExpiryParams,
    ) -> SignalFrame:
        fd = indicators["fd_close"]
        improvement = fd.diff() > 0
        entry = (
            indicators["post_expiry"]
            & improvement
            & improvement.shift(1).fillna(False)
            & (fd < fd.shift(5))
        ).fillna(False)

        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        in_position = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if in_position:
                if i == entry_bar:
                    entry_price = opens[i]
                held = i - entry_bar + 1
                target_hit = (
                    np.isfinite(entry_price)
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held >= params.max_hold:
                    in_position = False
                    signals[i] = 0
                else:
                    signals[i] = 1
            elif bool(entry.iloc[i]):
                in_position = True
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
