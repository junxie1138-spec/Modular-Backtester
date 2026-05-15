from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    atr_window: int = 14
    comp_threshold: float = 0.20
    hold_bars: int = 4
    seasonal_min_obs: int = 6
    vol_target: float = 0.012


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778886875"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.atr_window) + 45

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_window), min_periods=int(params.atr_window)).mean()
        natr = atr / close.replace(0.0, np.nan)

        dom = pd.Series(data.index.day, index=data.index)

        seasonal_natr = natr.groupby(dom).transform(
            lambda s: s.shift(1).expanding().mean()
        )
        natr_obs = natr.groupby(dom).transform(
            lambda s: s.shift(1).expanding().count()
        )
        seasonal_ret = ret.groupby(dom).transform(
            lambda s: s.shift(1).expanding().mean()
        )

        denom = seasonal_natr.replace(0.0, np.nan)
        compression = ((seasonal_natr - natr) / denom).fillna(0.0)
        compression = compression.replace([np.inf, -np.inf], 0.0)

        out = pd.DataFrame(index=data.index)
        out["natr"] = natr
        out["seasonal_natr"] = seasonal_natr
        out["natr_obs"] = natr_obs.fillna(0.0)
        out["seasonal_ret"] = seasonal_ret
        out["compression"] = compression
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)

        compression = indicators["compression"].fillna(0.0)
        valid = indicators["natr_obs"].fillna(0.0) >= float(params.seasonal_min_obs)
        compressed = (compression > float(params.comp_threshold)) & valid
        confirm = compressed & compressed.shift(1).fillna(False)

        direction = np.sign(indicators["seasonal_ret"].fillna(0.0)).astype(int)

        entry = np.where(
            confirm.to_numpy() & (direction.to_numpy() != 0),
            direction.to_numpy(),
            0,
        ).astype(int)

        hold = max(1, int(params.hold_bars))
        position = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if entry[i] != 0:
                d = int(entry[i])
                end = min(i + hold, n)
                position[i:end] = d
                i = end
            else:
                i += 1

        natr = indicators["natr"].fillna(0.01).clip(lower=1e-4)
        size_raw = (float(params.vol_target) / natr).clip(0.3, 3.0)
        size = size_raw * (1.0 + compression.clip(0.0, 2.0))
        size = size.clip(0.1, 5.0).fillna(1.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["size"] = size.to_numpy()
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
