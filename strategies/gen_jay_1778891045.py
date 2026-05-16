from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    atr_pct_window: int = 60
    atr_pct_thresh: float = 0.35
    vol_window: int = 20
    vol_z_thresh: float = 1.5
    snr_thresh: float = 0.50
    breakout_window: int = 10
    trail_k: float = 2.0
    base_size: float = 0.95
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778891045"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_period + params.atr_pct_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        atr_pct = atr.rolling(params.atr_pct_window).rank(pct=True)

        vol_mean = volume.rolling(params.vol_window).mean()
        vol_std = volume.rolling(params.vol_window).std()
        vol_z = (volume - vol_mean) / vol_std.where(vol_std > 0, other=np.nan)

        bar_range = (high - low).where(high > low, other=np.nan)
        body = (close - open_).abs()
        bar_snr = body / bar_range

        rolling_high = close.shift(1).rolling(params.breakout_window).max()

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["atr_pct"] = atr_pct
        ind["vol_z"] = vol_z
        ind["bar_snr"] = bar_snr
        ind["rolling_high"] = rolling_high
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        atr_pct = indicators["atr_pct"].values
        vol_z = indicators["vol_z"].values
        bar_snr = indicators["bar_snr"].values
        rolling_high = indicators["rolling_high"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size, dtype=float)

        in_trade = False
        hwm = 0.0
        entry_size = params.base_size

        for i in range(n):
            c = close[i]
            a = atr[i]

            if in_trade:
                if np.isfinite(c) and c > hwm:
                    hwm = c
                stop_triggered = (
                    np.isfinite(a)
                    and a > 0.0
                    and np.isfinite(hwm)
                    and np.isfinite(c)
                    and c < hwm - params.trail_k * a
                )
                if stop_triggered:
                    in_trade = False
                    hwm = 0.0
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                ap = atr_pct[i]
                vz = vol_z[i]
                sn = bar_snr[i]
                rh = rolling_high[i]

                entry_ok = (
                    np.isfinite(ap)
                    and ap < params.atr_pct_thresh
                    and np.isfinite(vz)
                    and vz > params.vol_z_thresh
                    and np.isfinite(sn)
                    and sn > params.snr_thresh
                    and np.isfinite(rh)
                    and np.isfinite(c)
                    and c > rh
                )
                if entry_ok:
                    in_trade = True
                    hwm = c
                    compression_factor = 1.0 - ap
                    vol_factor = min(vz / 4.0, 1.0)
                    raw = (
                        params.base_size
                        * (0.5 + 0.5 * compression_factor)
                        * (0.5 + 0.5 * vol_factor)
                    )
                    entry_size = min(raw, params.max_size)
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
