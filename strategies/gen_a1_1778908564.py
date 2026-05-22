from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    peak_w: int = 60
    rank_w: int = 120
    entry_q: float = 0.20
    shock_w: int = 10
    quiet_bars: int = 4
    regime_w: int = 200
    atr_w: int = 14
    k_stop: float = 2.5
    max_hold: int = 5
    vol_w: int = 20
    target_vol: float = 0.012
    size_min: float = 0.25
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_a1_1778908564"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        p = params
        return int(
            max(
                p.regime_w,
                p.peak_w + p.rank_w,
                p.peak_w + p.shock_w + p.quiet_bars,
                p.vol_w,
                p.atr_w,
            )
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        p = params
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        out = pd.DataFrame(index=data.index)

        # Drawdown depth relative to a rolling peak (<= 0).
        peak = close.rolling(p.peak_w).max()
        dd = close / peak - 1.0
        out["drawdown"] = dd

        # Percentile threshold (twist): entry_q-th quantile of the drawdown's
        # own recent distribution -- a deep, self-calibrating level.
        out["dd_threshold"] = dd.rolling(p.rank_w).quantile(p.entry_q)

        # Shockwave front: a fresh shock_w-bar drawdown low. The front has
        # dissipated when no new low has printed for quiet_bars bars.
        roll_min = dd.rolling(p.shock_w).min()
        is_new_low = dd <= (roll_min + 1e-9)
        new_low_count = is_new_low.rolling(p.quiet_bars).sum()
        out["dissipated"] = (new_low_count <= 0.0).astype(float)

        # Bull-regime filter.
        sma = close.rolling(p.regime_w).mean()
        out["regime"] = (close > sma).astype(float)

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
        out["atr"] = tr.rolling(p.atr_w).mean()

        # Volatility-targeting position size.
        ret = close.pct_change()
        rv = ret.rolling(p.vol_w).std()
        rv = rv.replace(0.0, np.nan)
        size = p.target_vol / rv
        size = size.clip(lower=p.size_min, upper=p.size_max)
        size = size.fillna(p.size_min)
        out["size"] = size

        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        p = params
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        thr = indicators["dd_threshold"].to_numpy(dtype=float)
        diss = indicators["dissipated"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        valid = (
            ~np.isnan(dd)
            & ~np.isnan(thr)
            & ~np.isnan(atr)
            & ~np.isnan(diss)
            & ~np.isnan(regime)
        )
        entry_raw = (
            valid
            & (regime > 0.5)
            & (diss > 0.5)
            & (dd <= thr)
            & (atr > 0.0)
        )

        sig = np.zeros(n, dtype=int)
        in_pos = False
        stop_level = 0.0
        held = 0

        for i in range(n):
            if in_pos:
                held += 1
                if close[i] < stop_level or held >= p.max_hold:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                if entry_raw[i]:
                    in_pos = True
                    stop_level = close[i] - p.k_stop * atr[i]
                    held = 0
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].to_numpy(dtype=float)
        size = np.where(np.isfinite(size) & (size > 0.0), size, p.size_min)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
