from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilReleaseParams:
    compression_window: int = 20
    atr_stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779652654"

    @classmethod
    def params_type(cls):
        return CoilReleaseParams

    def warmup_bars(self, params: CoilReleaseParams) -> int:
        return 200 + int(params.compression_window) + 5

    def indicators(self, data: pd.DataFrame, params: CoilReleaseParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        cw = max(int(params.compression_window), 2)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_long = true_range.rolling(200, min_periods=200).mean()
        atr_short = true_range.rolling(cw, min_periods=cw).mean()

        roll_high = high.rolling(cw, min_periods=cw).max()
        roll_low = low.rolling(cw, min_periods=cw).min()
        envelope = roll_high - roll_low

        denom = (atr_long * cw).replace(0, np.nan)
        compression_ratio = envelope / denom

        vol_mean = volume.rolling(cw, min_periods=cw).mean()

        out = pd.DataFrame(index=data.index)
        out["true_range"] = true_range
        out["atr_long"] = atr_long
        out["atr_short"] = atr_short
        out["roll_high_prev"] = roll_high.shift(1)
        out["compression_ratio_prev"] = compression_ratio.shift(1)
        out["vol_mean_prev"] = vol_mean.shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilReleaseParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        volume = data["volume"].to_numpy(dtype=float)
        roll_high_prev = indicators["roll_high_prev"].to_numpy(dtype=float)
        compression_prev = indicators["compression_ratio_prev"].to_numpy(dtype=float)
        vol_mean_prev = indicators["vol_mean_prev"].to_numpy(dtype=float)
        atr_short = indicators["atr_short"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)

        compression_threshold = 0.40
        volume_mult = 1.20
        k = float(params.atr_stop_mult)
        max_hold = 5

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                stop = entry_price - k * entry_atr
                if close[i] <= stop or bars_held >= max_hold:
                    in_pos = False
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                    continue
                raw_signal[i] = 1
                continue

            rh = roll_high_prev[i]
            cr = compression_prev[i]
            vm = vol_mean_prev[i]
            ats = atr_short[i]

            if not (np.isfinite(rh) and np.isfinite(cr) and np.isfinite(vm) and np.isfinite(ats)):
                continue
            if ats <= 0.0 or vm <= 0.0:
                continue

            loaded = cr < compression_threshold
            breakout = close[i] > rh
            confirmed = volume[i] > volume_mult * vm

            if loaded and breakout and confirmed:
                raw_signal[i] = 1
                in_pos = True
                entry_price = close[i]
                entry_atr = ats
                bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
