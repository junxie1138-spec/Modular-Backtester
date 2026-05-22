from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenParams:
    ma_length: int = 50
    zscore_lookback: int = 50
    z_upper: float = -0.3
    z_lower: float = -2.0
    eom_days: int = 3
    bom_days: int = 2
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[GenParams]):
    strategy_id = "gen_1778828977"

    @classmethod
    def params_type(cls):
        return GenParams

    @classmethod
    def warmup_bars(cls, params: GenParams) -> int:
        return int(max(params.ma_length, params.zscore_lookback)) + 5

    def indicators(self, data: pd.DataFrame, params: GenParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        ma_len = int(params.ma_length)
        z_len = int(params.zscore_lookback)

        ma = close.rolling(ma_len, min_periods=ma_len).mean()
        std = close.rolling(z_len, min_periods=z_len).std()
        std_safe = std.where(std > 0.0, np.nan)
        zscore = (close - ma) / std_safe

        idx = data.index
        month_key = pd.Series(
            np.asarray(idx.year, dtype=np.int64) * 12 + np.asarray(idx.month, dtype=np.int64),
            index=idx,
        )
        grouped = month_key.groupby(month_key)
        pos_from_start = grouped.cumcount()
        pos_from_end = grouped.cumcount(ascending=False)

        bom_n = int(params.bom_days)
        eom_n = int(params.eom_days)
        bom_window = (pos_from_start < bom_n)
        eom_window = (pos_from_end < eom_n)
        seasonal = (bom_window | eom_window).astype(float)

        out = pd.DataFrame(
            {
                "ma": ma,
                "zscore": zscore,
                "seasonal": seasonal,
                "bom_window": bom_window.astype(float),
                "eom_window": eom_window.astype(float),
            },
            index=idx,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenParams,
    ) -> SignalFrame:
        zscore = indicators["zscore"]
        seasonal = indicators["seasonal"].fillna(0.0) > 0.0

        z_upper = float(params.z_upper)
        z_lower = float(params.z_lower)
        elastic = (zscore <= z_upper) & (zscore >= z_lower)
        elastic = elastic.fillna(False)

        raw_signal = (seasonal & elastic).astype(int)

        hold = max(1, int(params.hold_bars))
        held = raw_signal.rolling(hold, min_periods=1).max().fillna(0).astype(int)

        signal = held.shift(1).fillna(0).astype(int)

        df = pd.DataFrame(
            {
                "signal": signal,
                "size": pd.Series(1.0, index=data.index),
            },
            index=data.index,
        )

        return SignalFrame(data=df, signal_column="signal", size_column="size")
