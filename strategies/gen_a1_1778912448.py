from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InsideBarCoilParams:
    inside_window: int = 20
    rank_window: int = 120
    rank_threshold: float = 0.80
    atr_period: int = 14
    trail_k: float = 3.0
    ma_period: int = 200
    use_trend_gate: bool = True
    spike_atr_mult: float = 2.5
    refractory_bars: int = 5


class GeneratedStrategy(BaseStrategy[InsideBarCoilParams]):
    strategy_id = "gen_a1_1778912448"

    @classmethod
    def params_type(cls) -> type[InsideBarCoilParams]:
        return InsideBarCoilParams

    @staticmethod
    def warmup_bars(params: InsideBarCoilParams) -> int:
        return int(
            max(
                params.ma_period,
                params.inside_window + params.rank_window,
                params.atr_period,
            )
            + 2
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: InsideBarCoilParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True range and ATR (Wilder-style EWM).
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(
            alpha=1.0 / float(params.atr_period),
            adjust=False,
            min_periods=params.atr_period,
        ).mean()

        # Inside bar: current high/low fully contained within prior bar's range.
        inside = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float)
        inside_frac = inside.rolling(
            params.inside_window, min_periods=params.inside_window
        ).mean()
        # Compression rank: where today's inside-bar fraction sits in its history.
        inside_rank = inside_frac.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        up_close = (close > prev_close).astype(float)

        # Spike: an outsized single-bar close-to-close move relative to ATR.
        move = close.diff().abs()
        spike = (move > params.spike_atr_mult * atr).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["inside_rank"] = inside_rank
        ind["ma"] = ma
        ind["up_close"] = up_close
        ind["spike"] = spike
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InsideBarCoilParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        inside_rank = indicators["inside_rank"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        up_close = indicators["up_close"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)

        # Two-bar confirmation: two consecutive up-closes into the bar.
        two_bar_conf = np.zeros(n, dtype=bool)
        for i in range(2, n):
            two_bar_conf[i] = (up_close[i] == 1.0) and (up_close[i - 1] == 1.0)

        position = 0
        hwm = 0.0  # in-trade high-water mark of close
        refr = 0  # refractory countdown after a spike

        for i in range(n):
            # A spike opens a refractory lockout window.
            if spike[i] == 1.0:
                refr = params.refractory_bars

            rank_i = inside_rank[i]
            atr_i = atr[i]
            ma_i = ma[i]

            valid = not (np.isnan(rank_i) or np.isnan(atr_i))
            trend_ok = True
            if params.use_trend_gate:
                trend_ok = (not np.isnan(ma_i)) and (close[i] > ma_i)

            if position == 0:
                entry_ok = (
                    valid
                    and trend_ok
                    and two_bar_conf[i]
                    and rank_i >= params.rank_threshold
                    and refr == 0
                    and spike[i] == 0.0
                )
                if entry_ok:
                    position = 1
                    hwm = close[i]
            else:
                # Ratchet the high-water mark up, never down.
                if close[i] > hwm:
                    hwm = close[i]
                if not np.isnan(atr_i):
                    stop = hwm - params.trail_k * atr_i
                    if close[i] < stop:
                        position = 0

            signal[i] = position

            if refr > 0:
                refr -= 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
