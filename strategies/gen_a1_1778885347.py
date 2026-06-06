from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockFadeParams:
    range_window: int = 20
    shock_mult: float = 1.5
    clv_low: float = 0.35
    clv_high: float = 0.65
    atr_window: int = 14
    atr_stop_k: float = 2.5
    max_hold: int = 18
    size_base: float = 0.4
    size_scale: float = 1.2
    size_max: float = 1.0
    range_excess_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[RangeShockFadeParams]):
    """Mean-reversion: fade the intra-bar close location of range-expansion shocks.

    A bar whose high-low range is far above its rolling baseline is treated as an
    overreaction. The close location within that wide bar (CLV) tells which side
    overshot: a low CLV (close near the low) -> panic -> go long; a high CLV
    (close near the high) -> euphoria -> go short. Signal strength = range excess
    * close-location extremity, and that strength scales the position size. Each
    trade is closed by a FIXED ATR volatility-stop measured from the entry price,
    or by a max-hold time cap if reversion plays out slowly.
    """

    strategy_id = "gen_a1_1778885347"

    @classmethod
    def params_type(cls) -> type[RangeShockFadeParams]:
        return RangeShockFadeParams

    @staticmethod
    def warmup_bars(params: RangeShockFadeParams) -> int:
        return int(max(params.range_window, params.atr_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeShockFadeParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = (high - low)
        rng_np = rng.to_numpy()
        baseline = rng.rolling(params.range_window, min_periods=params.range_window).mean()
        range_ratio = rng / baseline.replace(0.0, np.nan)

        safe_rng = np.where(rng_np > 0.0, rng_np, 1.0)
        clv_np = np.where(rng_np > 0.0,
                          (close.to_numpy() - low.to_numpy()) / safe_rng,
                          0.5)
        clv = pd.Series(clv_np, index=data.index).clip(lower=0.0, upper=1.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        range_excess = (range_ratio - params.shock_mult).clip(lower=0.0, upper=params.range_excess_cap)

        clv_low = max(params.clv_low, 1e-6)
        clv_high = min(params.clv_high, 1.0 - 1e-6)
        long_loc = ((clv_low - clv) / clv_low).clip(lower=0.0, upper=1.0)
        short_loc = ((clv - clv_high) / (1.0 - clv_high)).clip(lower=0.0, upper=1.0)

        long_strength = (range_excess * long_loc).fillna(0.0)
        short_strength = (range_excess * short_loc).fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["range_ratio"] = range_ratio
        out["clv"] = clv
        out["atr"] = atr
        out["long_strength"] = long_strength
        out["short_strength"] = short_strength
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeShockFadeParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_s = indicators["long_strength"].to_numpy(dtype=float)
        short_s = indicators["short_strength"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        entry_size = 1.0
        bars_held = 0

        k = float(params.atr_stop_k)
        max_hold = int(params.max_hold)
        size_base = float(params.size_base)
        size_scale = float(params.size_scale)
        size_max = float(params.size_max)

        for i in range(n):
            if pos == 0:
                a = atr[i]
                if not np.isfinite(a) or a <= 0.0:
                    continue
                ls = long_s[i]
                ss = short_s[i]
                if not np.isfinite(ls):
                    ls = 0.0
                if not np.isfinite(ss):
                    ss = 0.0
                if ls > 0.0 and ls >= ss:
                    pos = 1
                    entry_price = close[i]
                    entry_atr = a
                    entry_size = min(size_max, size_base + size_scale * ls)
                    bars_held = 0
                    signal[i] = 1
                    size[i] = entry_size
                elif ss > 0.0:
                    pos = -1
                    entry_price = close[i]
                    entry_atr = a
                    entry_size = min(size_max, size_base + size_scale * ss)
                    bars_held = 0
                    signal[i] = -1
                    size[i] = entry_size
            else:
                bars_held += 1
                exit_now = False
                if pos == 1:
                    if close[i] < entry_price - k * entry_atr:
                        exit_now = True
                    elif bars_held >= max_hold:
                        exit_now = True
                else:
                    if close[i] > entry_price + k * entry_atr:
                        exit_now = True
                    elif bars_held >= max_hold:
                        exit_now = True
                if exit_now:
                    signal[i] = 0
                    size[i] = entry_size
                    pos = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                else:
                    signal[i] = pos
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0).clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
