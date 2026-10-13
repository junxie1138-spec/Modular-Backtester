from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryPressureParams:
    diff_order: float = 0.4
    diff_window: int = 20
    profit_target: float = 0.008
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ExpiryPressureParams]):
    strategy_id = "gen_local_1791300739"

    @classmethod
    def params_type(cls) -> type[ExpiryPressureParams]:
        return ExpiryPressureParams

    @classmethod
    def warmup_bars(cls, params: ExpiryPressureParams) -> int:
        return params.diff_window + 1

    def indicators(self, data: pd.DataFrame, params: ExpiryPressureParams) -> pd.DataFrame:
        weights = np.empty(params.diff_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.diff_window):
            weights[lag] = weights[lag - 1] * (lag - 1 - params.diff_order) / lag

        fractional_close = sum(
            weights[lag] * data["close"].shift(lag)
            for lag in range(params.diff_window)
        )
        valid = pd.Series(np.arange(len(data)) >= params.diff_window, index=data.index)
        fractional_step = fractional_close.diff().where(valid, 0.0).fillna(0.0)
        weak_two_bars = (fractional_step < 0) & (fractional_step.shift(1).fillna(0.0) < 0)
        expiry_tuesday = (
            (data.index.weekday == 1)
            & (data.index.day >= 12)
            & (data.index.day <= 18)
        )
        return pd.DataFrame(
            {
                "fractional_close": fractional_close.fillna(0.0),
                "fractional_step": fractional_step,
                "weak_two_bars": weak_two_bars.fillna(False),
                "expiry_tuesday": expiry_tuesday,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpiryPressureParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = 0.0

        for i in range(len(data)):
            invested = i > 0 and raw[i - 1] == 1
            if invested:
                if i == 1 or raw[i - 2] == 0:
                    entry_bar = i
                    entry_price = opens[i]
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.max_hold_bars
                raw[i] = 0 if target_hit or time_stop else 1
            elif indicators["expiry_tuesday"].iat[i] and indicators["weak_two_bars"].iat[i]:
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
