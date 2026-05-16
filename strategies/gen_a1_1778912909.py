from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    range_window: int = 20
    snr_std_window: int = 20
    rank_window: int = 60
    vol_window: int = 60
    atr_window: int = 14
    entry_pct: float = 0.90
    vol_pct: float = 0.70
    atr_mult: float = 2.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778912909"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        base = (
            params.range_window
            + params.snr_std_window
            + params.rank_window
            + 5
        )
        return int(max(base, params.vol_window + 2, params.atr_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        n = max(int(params.range_window), 2)
        m = max(int(params.snr_std_window), 2)
        w = max(int(params.rank_window), 5)
        v = max(int(params.vol_window), 5)
        p = max(int(params.atr_window), 2)

        # Relative position of close inside the rolling high-low range.
        rng_high = high.rolling(n).max()
        rng_low = low.rolling(n).min()
        denom = (rng_high - rng_low).replace(0.0, np.nan)
        relpos = (close - rng_low) / denom
        relpos = relpos.clip(lower=0.0, upper=1.0)
        ind["relpos"] = relpos

        # One-bar change in relative position (positional velocity).
        d_relpos = relpos.diff()
        ind["d_relpos"] = d_relpos

        # Signal-to-noise: positional move divided by its own recent dispersion.
        noise = d_relpos.rolling(m).std().replace(0.0, np.nan)
        snr = d_relpos / noise
        ind["snr"] = snr

        # Percentile rank of |SNR| over its own recent distribution (twist).
        snr_abs = snr.abs()
        ind["snr_rank"] = snr_abs.rolling(w).rank(pct=True)

        # Volume confirmation as a percentile rank, not a fixed level.
        ind["vol_rank"] = volume.rolling(v).rank(pct=True)

        # ATR for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(p).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        close = data["close"].to_numpy(dtype=float)
        d_relpos = indicators["d_relpos"].fillna(0.0).to_numpy(dtype=float)
        snr_rank = indicators["snr_rank"].fillna(0.0).to_numpy(dtype=float)
        vol_rank = indicators["vol_rank"].fillna(0.0).to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        entry_pct = float(params.entry_pct)
        vol_pct = float(params.vol_pct)
        k = float(params.atr_mult)
        max_hold = max(int(params.max_hold), 1)

        # Raw volume-confirmed, SNR-filtered, percentile-gated entry direction.
        clean = snr_rank >= entry_pct
        confirmed = vol_rank >= vol_pct
        raw_long = clean & confirmed & (d_relpos > 0.0)
        raw_short = clean & confirmed & (d_relpos < 0.0)
        raw_dir = np.where(raw_long, 1, np.where(raw_short, -1, 0))

        sig = np.zeros(n, dtype=int)

        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if pos == 0:
                a = atr[i]
                if raw_dir[i] != 0 and np.isfinite(a) and a > 0.0:
                    pos = int(raw_dir[i])
                    entry_price = close[i]
                    entry_atr = a
                    bars_held = 0
            else:
                bars_held += 1
                exit_now = False
                stop = k * entry_atr
                if pos == 1 and close[i] < entry_price - stop:
                    exit_now = True
                elif pos == -1 and close[i] > entry_price + stop:
                    exit_now = True
                if bars_held >= max_hold:
                    exit_now = True
                if exit_now:
                    pos = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
            sig[i] = pos

        df["signal"] = sig
        df["size"] = 1.0

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
