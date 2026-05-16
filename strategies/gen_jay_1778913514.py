from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSNRTrendParams:
    gap_window: int = 10
    pct_lookback: int = 252
    entry_pct: float = 0.82
    snr_min_pct: float = 0.65
    atr_window: int = 14
    trail_k: float = 2.0
    trend_window: int = 50


class GeneratedStrategy(BaseStrategy["GapSNRTrendParams"]):
    strategy_id = "gen_jay_1778913514"

    @classmethod
    def params_type(cls):
        return GapSNRTrendParams

    @staticmethod
    def warmup_bars(params: GapSNRTrendParams) -> int:
        return params.pct_lookback + params.gap_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapSNRTrendParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        gap = data["open"] - close.shift(1)

        gap_sum = gap.rolling(params.gap_window).sum()
        abs_gap_sum = gap.abs().rolling(params.gap_window).sum()
        gap_snr = gap_sum.abs() / abs_gap_sum.where(abs_gap_sum > 0, other=np.nan)

        gap_mom_pct = gap_sum.rolling(params.pct_lookback).rank(pct=True)
        snr_pct = gap_snr.rolling(params.pct_lookback).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ma = close.rolling(params.trend_window).mean()
        above_ma = (close > ma).astype(float).where(ma.notna(), other=np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["gap_mom_pct"] = gap_mom_pct
        ind["snr_pct"] = snr_pct
        ind["atr"] = atr
        ind["above_ma"] = above_ma

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSNRTrendParams,
    ) -> SignalFrame:
        n = len(data)
        close_arr = data["close"].values

        gap_mom_pct = indicators["gap_mom_pct"].values
        snr_pct = indicators["snr_pct"].values
        atr_arr = indicators["atr"].values
        above_ma = indicators["above_ma"].values

        signal = np.zeros(n, dtype=int)

        position = 0
        hwm = 0.0
        lwm = 0.0

        for i in range(n):
            c = close_arr[i]

            if (
                np.isnan(gap_mom_pct[i])
                or np.isnan(snr_pct[i])
                or np.isnan(atr_arr[i])
                or np.isnan(above_ma[i])
            ):
                signal[i] = 0
                continue

            trail = params.trail_k * atr_arr[i]

            if position == 1:
                if c > hwm:
                    hwm = c
                if c < hwm - trail:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

            elif position == -1:
                if c < lwm:
                    lwm = c
                if c > lwm + trail:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = -1

            else:
                snr_ok = snr_pct[i] > params.snr_min_pct
                long_ok = (
                    gap_mom_pct[i] > params.entry_pct
                    and snr_ok
                    and above_ma[i] == 1.0
                )
                short_ok = (
                    gap_mom_pct[i] < (1.0 - params.entry_pct)
                    and snr_ok
                    and above_ma[i] == 0.0
                )

                if long_ok:
                    position = 1
                    hwm = c
                    signal[i] = 1
                elif short_ok:
                    position = -1
                    lwm = c
                    signal[i] = -1
                else:
                    signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
