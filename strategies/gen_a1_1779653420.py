from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenParams:
    lookback: int = 5
    jam_z: float = 1.0


class GeneratedStrategy(BaseStrategy[GenParams]):
    strategy_id = "gen_a1_1779653420"

    @classmethod
    def params_type(cls):
        return GenParams

    @classmethod
    def warmup_bars(cls, params: GenParams) -> int:
        return int(max(20, int(params.lookback)) + 1)

    def indicators(self, data: pd.DataFrame, params: GenParams) -> pd.DataFrame:
        L = int(params.lookback)
        close = data["close"].astype(float)
        rng = (data["high"].astype(float) - data["low"].astype(float))
        rng = rng.where(rng > 0.0, np.nan)
        density = data["volume"].astype(float) / rng
        d_mean = density.rolling(20, min_periods=20).mean()
        d_std = density.rolling(20, min_periods=20).std(ddof=0)
        d_std_safe = d_std.where(d_std > 0.0, np.nan)
        density_z = (density - d_mean) / d_std_safe
        ret_L = close.pct_change(L)
        return pd.DataFrame(
            {
                "density_z": density_z,
                "ret_L": ret_L,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenParams,
    ) -> SignalFrame:
        density_z = indicators["density_z"]
        ret_L = indicators["ret_L"]
        threshold = float(params.jam_z)
        entry = (density_z > threshold) & (ret_L < 0.0)
        raw = entry.fillna(False).astype(int)
        shifted = raw.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)
        df = pd.DataFrame(
            {
                "signal": shifted.values,
                "size": size.values,
            },
            index=data.index,
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
