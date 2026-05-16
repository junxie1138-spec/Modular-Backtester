from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NeapTideParams:
    comp_len: int = 40
    comp_thresh: float = 0.25
    chan_len: int = 20
    low_node_thresh: float = 0.25
    atr_len: int = 14
    init_stop_k: float = 2.0
    be_trigger: float = 0.01
    trail_k: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[NeapTideParams]):
    strategy_id = "gen_a1_1778896013"

    @classmethod
    def params_type(cls):
        return NeapTideParams

    @staticmethod
    def warmup_bars(params: NeapTideParams) -> int:
        return int(max(params.comp_len, params.chan_len, params.atr_len)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: NeapTideParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        # rolling percentile rank of the current true range within its window
        range_pct = tr.rolling(
            params.comp_len, min_periods=params.comp_len
        ).rank(pct=True)

        roll_high = high.rolling(params.chan_len, min_periods=params.chan_len).max()
        roll_low = low.rolling(params.chan_len, min_periods=params.chan_len).min()
        span = (roll_high - roll_low).replace(0.0, np.nan)
        chan_pos = (close - roll_low) / span

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["range_pct"] = range_pct
        out["chan_pos"] = chan_pos
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: NeapTideParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        range_pct = indicators["range_pct"].to_numpy(dtype=float)
        chan_pos = indicators["chan_pos"].to_numpy(dtype=float)

        n = len(data)

        valid = ~(np.isnan(range_pct) | np.isnan(chan_pos) | np.isnan(atr))
        compressed = np.where(valid, range_pct <= params.comp_thresh, False)
        at_low = np.where(valid, chan_pos <= params.low_node_thresh, False)
        raw = compressed & at_low & valid

        # two-bar confirmation: the entry condition must hold on this bar
        # and the immediately preceding bar before any long is taken.
        prior = np.zeros(n, dtype=bool)
        if n > 1:
            prior[1:] = raw[:-1]
        confirmed = raw & prior

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        high_water = 0.0
        be_armed = False
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                if high[i] > high_water:
                    high_water = high[i]
                a = atr[i]
                # breakeven: once price has run +be_trigger, lift stop to entry
                if (
                    not be_armed
                    and high_water >= entry_price * (1.0 + params.be_trigger)
                ):
                    if entry_price > stop:
                        stop = entry_price
                    be_armed = True
                # trail: after breakeven arms, ratchet stop up by k*ATR only
                if be_armed and not np.isnan(a):
                    trail = high_water - params.trail_k * a
                    if trail > stop:
                        stop = trail
                exit_now = (low[i] <= stop) or (bars_held >= params.max_hold)
                if exit_now:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if confirmed[i] and not np.isnan(atr[i]):
                    in_pos = True
                    entry_price = close[i]
                    high_water = high[i]
                    stop = entry_price - params.init_stop_k * atr[i]
                    be_armed = False
                    bars_held = 0
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
