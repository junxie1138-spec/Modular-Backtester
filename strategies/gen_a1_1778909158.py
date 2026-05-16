from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilSpringParams:
    atr_period: int = 14
    k_atr: float = 2.0
    perc_window: int = 120
    perc_q: float = 0.85
    min_coil: int = 2
    sma_long: int = 100
    max_hold: int = 2
    base_size: float = 1.0
    size_cap: float = 1.8
    size_floor: float = 0.5


class GeneratedStrategy(BaseStrategy[CoilSpringParams]):
    """Momentum: enter the upward release of a percentile-deep volatility coil.

    The signal primitive is a consecutive-streak count of bars whose true
    range contracts versus the prior bar. When that live streak reaches the
    top percentile of its own trailing distribution, the spring is fully
    wound. Entry fires on the first bar that breaks the coil with range
    re-expansion and an up close, inside an uptrend regime. Exit is a fixed
    ATR volatility-stop set at entry, plus a 1-2 day max hold.
    """

    strategy_id = "gen_a1_1778909158"

    @classmethod
    def params_type(cls):
        return CoilSpringParams

    def warmup_bars(self, params: CoilSpringParams) -> int:
        return int(max(params.atr_period, params.perc_window, params.sma_long)) + 2

    def indicators(self, data: pd.DataFrame, params: CoilSpringParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        hl = (high - low).abs()
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        tr = tr.fillna(hl)

        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        # Consecutive-streak count of range-contraction bars (the coil).
        contracting = (tr < tr.shift(1)).fillna(False)
        blocks = (~contracting).cumsum()
        coil_streak = contracting.groupby(blocks).cumsum().astype(float)

        # Twist: percentile threshold instead of a fixed streak level.
        coil_thresh = coil_streak.rolling(
            params.perc_window, min_periods=params.perc_window
        ).quantile(params.perc_q)

        sma = close.rolling(params.sma_long, min_periods=params.sma_long).mean()

        coil_prev = coil_streak.shift(1)
        thresh_prev = coil_thresh.shift(1)
        tr_expand = tr > tr.shift(1)
        up_close = close > close.shift(1)
        in_trend = close > sma

        entry = (
            (coil_prev >= thresh_prev)
            & (coil_prev >= float(params.min_coil))
            & tr_expand
            & up_close
            & in_trend
        ).fillna(False)

        # Spring tension: deeper coils relative to threshold size up larger.
        denom = thresh_prev.where(thresh_prev > 0.0, np.nan)
        ratio = coil_prev / denom
        elastic = (float(params.base_size) * ratio).clip(
            lower=float(params.size_floor), upper=float(params.size_cap)
        )
        elastic = elastic.fillna(float(params.base_size)).clip(
            lower=float(params.size_floor), upper=float(params.size_cap)
        )

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["atr"] = atr
        out["coil_streak"] = coil_streak
        out["coil_thresh"] = coil_thresh
        out["sma"] = sma
        out["entry"] = entry.astype(float)
        out["elastic_size"] = elastic
        return out

    def generate_signals(self, data, indicators, ctx, params):
        idx = data.index
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_ok = indicators["entry"].to_numpy(dtype=float)
        elastic = indicators["elastic_size"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_pos = False
        stop = 0.0
        bars_held = 0
        current_size = float(params.base_size)
        max_hold = int(params.max_hold)
        k = float(params.k_atr)

        for i in range(n):
            if in_pos:
                bars_held += 1
                exit_now = (
                    not np.isfinite(close[i])
                    or close[i] < stop
                    or bars_held >= max_hold
                )
                if exit_now:
                    in_pos = False
                    raw[i] = 0
                    bars_held = 0
                    continue
                raw[i] = 1
                size[i] = current_size
                continue

            if (
                entry_ok[i] >= 0.5
                and np.isfinite(atr[i])
                and atr[i] > 0.0
                and np.isfinite(close[i])
            ):
                in_pos = True
                # Fixed volatility-stop: locked at entry, never trailed.
                stop = close[i] - k * atr[i]
                bars_held = 0
                if np.isfinite(elastic[i]) and elastic[i] > 0.0:
                    current_size = float(elastic[i])
                else:
                    current_size = float(params.base_size)
                raw[i] = 1
                size[i] = current_size
            else:
                raw[i] = 0

        df = pd.DataFrame(index=idx)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(raw, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = (
            pd.Series(size, index=idx).shift(1).fillna(1.0).clip(lower=0.01)
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
