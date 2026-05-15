from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolSNRParams:
    vol_window: int = 10
    baseline_window: int = 63
    snr_window: int = 10
    vol_compression_thresh: float = 0.85
    snr_entry_thresh: float = 0.15
    snr_scale_cap: float = 0.50


class GeneratedStrategy(BaseStrategy["VolSNRParams"]):
    strategy_id = "gen_jay_1778884744"

    @classmethod
    def params_type(cls):
        return VolSNRParams

    @staticmethod
    def warmup_bars(params: VolSNRParams) -> int:
        return params.baseline_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolSNRParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        rolling_vol = ret.rolling(params.vol_window).std()
        baseline_vol = ret.rolling(params.baseline_window).std()
        vol_ratio = rolling_vol / baseline_vol.where(baseline_vol > 0, np.nan)

        snr_mean = ret.rolling(params.snr_window).mean()
        snr_std = ret.rolling(params.snr_window).std()
        snr = snr_mean / snr_std.where(snr_std > 0, np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["vol_ratio"] = vol_ratio
        ind["snr"] = snr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolSNRParams,
    ) -> SignalFrame:
        vol_ratio = indicators["vol_ratio"].values
        snr = indicators["snr"].values

        n = len(data)
        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_pos = False
        for i in range(n):
            vr = vol_ratio[i]
            sr = snr[i]
            vr_ok = np.isfinite(vr)
            sr_ok = np.isfinite(sr)

            entry = vr_ok and sr_ok and (vr < params.vol_compression_thresh) and (sr > params.snr_entry_thresh)
            exit_ = (not vr_ok) or (not sr_ok) or (vr >= params.vol_compression_thresh) or (sr <= 0.0)

            if not in_pos:
                if entry:
                    in_pos = True
                    signal[i] = 1
                    size[i] = min(sr / params.snr_scale_cap, 1.0)
            else:
                if exit_:
                    in_pos = False
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = min(sr / params.snr_scale_cap, 1.0)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
