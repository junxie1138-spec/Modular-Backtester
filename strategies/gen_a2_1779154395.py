from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    sma_window: int = 50
    std_window: int = 50
    entry_z: float = -1.0
    half_len: int = 10
    profit_target: float = 0.04
    time_stop: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779154395"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.sma_window + params.std_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        sma = close.rolling(params.sma_window, min_periods=params.sma_window).mean()
        dist = close - sma
        dstd = dist.rolling(params.std_window, min_periods=params.std_window).std()
        z = dist / dstd.replace(0.0, np.nan)

        idx = data.index
        month = pd.PeriodIndex(idx, freq="M")
        tdom = pd.Series(np.arange(len(idx)), index=idx).groupby(month).cumcount() + 1

        out = pd.DataFrame(index=idx)
        out["z"] = z.astype(float)
        out["tdom"] = tdom.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        tdom = indicators["tdom"].to_numpy(dtype=float)
        raw = np.zeros(n, dtype=np.int64)

        pt = float(params.profit_target)
        ts = int(params.time_stop)
        half = float(params.half_len)
        ez = float(params.entry_z)

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                ret = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if ret >= pt or bars_held >= ts:
                    raw[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                else:
                    raw[i] = 1
                continue

            if i < 2:
                continue
            z0, z1, z2 = z[i], z[i - 1], z[i - 2]
            if np.isnan(z0) or np.isnan(z1) or np.isnan(z2):
                continue

            seasonal = tdom[i] <= half
            confirmed = (z0 > z1) and (z1 > z2)
            loaded = z0 < ez
            if seasonal and confirmed and loaded:
                raw[i] = 1
                in_pos = True
                entry_price = close[i]
                bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
