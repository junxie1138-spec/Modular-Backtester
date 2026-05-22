from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    clv_window: int = 5
    fast_range_window: int = 5
    slow_range_window: int = 20
    atr_window: int = 14
    loc_window: int = 20
    clv_low_thresh: float = 0.42
    contraction_thresh: float = 0.92
    snr_thresh: float = 0.6
    k_atr_stop: float = 2.5
    max_hold: int = 4
    base_size: float = 0.4
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778894531"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        longest = max(
            params.slow_range_window,
            params.loc_window,
            params.atr_window,
            params.fast_range_window,
            params.clv_window,
        )
        return int(longest + params.clv_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = high - low
        out["rng"] = rng

        # close-location-value: where the close finishes inside the day's range
        denom = rng.replace(0.0, np.nan)
        clv = ((close - low) / denom).fillna(0.5).clip(0.0, 1.0)
        out["clv"] = clv

        out["clv_avg"] = clv.rolling(params.clv_window, min_periods=params.clv_window).mean()
        out["clv_std"] = clv.rolling(params.clv_window, min_periods=params.clv_window).std()

        out["atr"] = rng.rolling(params.atr_window, min_periods=params.atr_window).mean()

        rng_fast = rng.rolling(params.fast_range_window, min_periods=params.fast_range_window).mean()
        rng_slow = rng.rolling(params.slow_range_window, min_periods=params.slow_range_window).mean()
        out["contraction"] = rng_fast / rng_slow.replace(0.0, np.nan)

        out["loc_ma"] = close.rolling(params.loc_window, min_periods=params.loc_window).mean()

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        clv_avg = indicators["clv_avg"].to_numpy(dtype=float)
        clv_std = indicators["clv_std"].to_numpy(dtype=float)
        contraction = indicators["contraction"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        loc_ma = indicators["loc_ma"].to_numpy(dtype=float)

        eps = 1e-9

        # depth: how far average close-location sits below the bearish threshold
        depth = (params.clv_low_thresh - clv_avg) / max(params.clv_low_thresh, eps)
        depth = np.clip(np.nan_to_num(depth, nan=0.0), 0.0, 1.0)

        # signal-to-noise: clean bearish-close-location reading vs its own jitter
        std_filled = np.nan_to_num(clv_std, nan=1.0)
        snr = depth / (std_filled + 0.05)
        snr_norm = np.clip(snr / 2.0, 0.0, 1.0)

        # dryness: intrabar range contraction below its longer-term baseline
        dryness = np.clip(1.0 - np.nan_to_num(contraction, nan=1.0), 0.0, 0.5) / 0.5

        # signal-scaled position strength (twist): depth driven, modulated by
        # range dryness and signal-to-noise quality
        strength = depth * (0.5 + 0.3 * dryness + 0.2 * snr_norm)
        strength = np.clip(strength, 0.0, 1.0)
        pos_size = params.base_size + (params.max_size - params.base_size) * strength
        pos_size = np.clip(pos_size, 0.05, params.max_size)

        valid = ~np.isnan(clv_avg) & ~np.isnan(contraction) & ~np.isnan(atr) & ~np.isnan(loc_ma)
        in_dip = close < np.nan_to_num(loc_ma, nan=np.inf)
        clv_ok = clv_avg < params.clv_low_thresh
        contraction_ok = contraction < params.contraction_thresh
        snr_ok = snr > params.snr_thresh
        enter = (in_dip & clv_ok & contraction_ok & snr_ok & valid).astype(bool)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_stop = 0.0
        bars_held = 0
        held_size = params.base_size

        for i in range(n):
            if in_pos:
                bars_held += 1
                # fixed volatility-stop (entry price minus k*ATR, set at entry)
                if close[i] < entry_stop or bars_held >= params.max_hold:
                    in_pos = False
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = held_size
            else:
                if enter[i]:
                    a = atr[i]
                    if not np.isfinite(a) or a <= 0.0:
                        a = 0.0
                    in_pos = True
                    bars_held = 0
                    entry_stop = close[i] - params.k_atr_stop * a
                    held_size = float(pos_size[i])
                    signal[i] = 1
                    size[i] = held_size
                else:
                    signal[i] = 0
                    size[i] = 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        df["size"] = df["size"].clip(lower=0.05)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
