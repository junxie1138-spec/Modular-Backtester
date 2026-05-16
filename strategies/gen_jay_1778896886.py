from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778896886"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        snr_window = max(5, params.ma_period // 4)
        return params.ma_period + snr_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        snr_window = max(5, params.ma_period // 4)

        ma = close.rolling(params.ma_period).mean()
        std = close.rolling(params.ma_period).std().clip(lower=1e-8)
        z = (close - ma) / std

        dz = z - z.shift(snr_window)
        z_noise = z.rolling(snr_window).std().clip(lower=1e-8)
        snr = dz.abs() / z_noise

        ind = pd.DataFrame(index=data.index)
        ind["z"] = z
        ind["dz"] = dz
        ind["snr"] = snr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)

        raw_entry = (
            (indicators["z"] < 0.0)
            & (indicators["dz"] > 0.0)
            & (indicators["snr"] > 1.0)
        ).fillna(False).values

        signal = np.zeros(n, dtype=int)
        exit_at = -1

        for i in range(n):
            if exit_at > 0 and i < exit_at:
                signal[i] = 1
            else:
                if exit_at > 0 and i >= exit_at:
                    exit_at = -1
                if raw_entry[i]:
                    signal[i] = 1
                    exit_at = i + params.hold_bars

        df = data[[]].copy()
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
