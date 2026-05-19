from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 10
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779155077"

    # Fixed structural constants - intentionally NOT tunable (twist: <=2 tunable params).
    _MEDIAN_WINDOW = 60
    _DENSITY_WINDOW = 40
    _DENSITY_THRESHOLD = 0.70

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        vw = max(2, int(params.vol_window))
        return vw + GeneratedStrategy._MEDIAN_WINDOW + GeneratedStrategy._DENSITY_WINDOW + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        vw = max(2, int(params.vol_window))

        # Realized volatility: rolling std of simple returns.
        ret = close.pct_change()
        vol = ret.rolling(vw).std()

        # Typical volatility level of the recent regime.
        vol_median = vol.rolling(GeneratedStrategy._MEDIAN_WINDOW).median()

        # 'Jam density': fraction of the last L bars whose vol sat below its median.
        # NaN-driven comparisons evaluate False, so warmup bars count as not-compressed.
        compressed = (vol < vol_median)
        jam_density = compressed.astype(float).rolling(GeneratedStrategy._DENSITY_WINDOW).mean()

        # Decompression front: vol crosses up through its rolling median.
        front = (vol > vol_median) & (vol.shift(1) <= vol_median.shift(1))

        # Net upward drift over the compression window confirms an up-resolution.
        uptrend = close > close.shift(vw)

        entry = (
            (jam_density >= GeneratedStrategy._DENSITY_THRESHOLD)
            & front.fillna(False)
            & uptrend.fillna(False)
        )

        ind = pd.DataFrame(index=data.index)
        ind["vol"] = vol
        ind["vol_median"] = vol_median
        ind["jam_density"] = jam_density.fillna(0.0)
        ind["entry"] = entry.fillna(False).astype(bool)
        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        entry = indicators["entry"].to_numpy()
        hold = max(1, int(params.hold_bars))

        # Fixed-bar exit: once long, hold exactly `hold` bars then flatten.
        # Path-dependent, so a bar-indexed loop is used.
        signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        held = 0
        for i in range(n):
            if in_pos:
                signal[i] = 1
                held += 1
                if held >= hold:
                    in_pos = False
                    held = 0
            elif bool(entry[i]):
                in_pos = True
                signal[i] = 1
                held = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
