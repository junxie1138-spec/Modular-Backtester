from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    percentile_window: int = 126
    energy_percentile: float = 0.8
    atr_window: int = 14
    stop_atr_mult: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791295534"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return max(16, params.atr_window) + params.percentile_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Each contrast is a Haar detail: recent half-window mean minus prior half-window mean.
        details = []
        for scale in (4, 8, 16):
            half = scale // 2
            mean = close.rolling(half, min_periods=half).mean()
            details.append(mean - mean.shift(half))
        energy = pd.concat(
            [(detail / atr.replace(0, np.nan)).clip(lower=0).pow(2) for detail in details],
            axis=1,
        ).min(axis=1, skipna=False)
        threshold = energy.rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.energy_percentile).shift(1)

        return pd.DataFrame({
            "atr": atr.fillna(0.0),
            "energy": energy.fillna(0.0),
            "threshold": threshold.fillna(0.0),
            "ready": threshold.notna(),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        energy = indicators["energy"]
        threshold = indicators["threshold"]
        entry = (
            indicators["ready"]
            & energy.gt(threshold)
            & energy.shift(1).le(threshold.shift(1)).fillna(False)
        )

        target = np.zeros(len(data), dtype=int)
        entry_bar = -1
        stop = 0.0
        for i in range(len(data)):
            if i > 0:
                if target[i - 1] == 1 and (i == 1 or target[i - 2] == 0):
                    entry_bar = i
                    stop = float(data["open"].iat[i]) - params.stop_atr_mult * float(indicators["atr"].iat[i - 1])
                elif target[i - 1] == 0:
                    entry_bar = -1

            if entry_bar >= 0:
                if float(data["close"].iat[i]) < stop or i - entry_bar + 1 >= params.hold_bars:
                    target[i] = 0
                else:
                    target[i] = 1
            elif bool(entry.iat[i]):
                target[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
