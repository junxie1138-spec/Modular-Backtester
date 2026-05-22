from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 60
    smooth_window: int = 20
    noise_window: int = 20
    snr_threshold: float = 1.0
    dd_size_k: float = 1.5
    base_size: float = 1.0
    max_size: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778887000"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.peak_window + params.smooth_window + params.noise_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        peak_w = max(2, int(params.peak_window))
        smooth_w = max(2, int(params.smooth_window))
        noise_w = max(2, int(params.noise_window))

        # Drawdown depth: fraction below the trailing peak, >= 0.
        peak = close.rolling(peak_w, min_periods=peak_w).max()
        dd = 1.0 - (close / peak)
        dd = dd.clip(lower=0.0)

        # The drawdown series treated as an oscillator vs its own mean.
        ref = dd.rolling(smooth_w, min_periods=smooth_w).mean()
        spread = dd - ref

        # Signal-to-noise: magnitude of the spread over its own volatility.
        noise = spread.rolling(noise_w, min_periods=noise_w).std()
        noise = noise.where(noise > 0.0, np.nan)
        snr = spread.abs() / noise

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["ref"] = ref
        out["spread"] = spread
        out["snr"] = snr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        spread = indicators["spread"].to_numpy(dtype=float)
        snr = indicators["snr"].to_numpy(dtype=float)
        dd = np.nan_to_num(indicators["dd"].to_numpy(dtype=float), nan=0.0)

        thr = float(params.snr_threshold)
        signal = np.zeros(n, dtype=int)
        pos = 0

        # Symmetric stop-and-reverse: the long entry (spread < 0, drawdown
        # healing below its mean) and the short entry (spread > 0, drawdown
        # deepening above its mean) are mirror images sharing one SNR gate.
        # A position is exited ONLY when the opposite entry condition fires;
        # when the SNR gate is closed the prior position is carried forward.
        for i in range(n):
            s = spread[i]
            r = snr[i]
            if np.isnan(s) or np.isnan(r):
                pos = 0
                signal[i] = 0
                continue
            if r >= thr:
                if s < 0.0:
                    pos = 1
                elif s > 0.0:
                    pos = -1
                # s == 0.0 exactly: carry the existing position.
            # r < thr: noise regime, carry the existing position.
            signal[i] = pos

        size = np.clip(
            float(params.base_size) * (1.0 + float(params.dd_size_k) * dd),
            0.5,
            float(params.max_size),
        )

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
