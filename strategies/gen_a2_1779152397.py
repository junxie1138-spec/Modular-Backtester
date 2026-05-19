from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_window: int = 60
    rank_window: int = 120
    arm_window: int = 20
    arm_lo: float = 0.15
    trigger_hi: float = 0.55
    ma_window: int = 200
    atr_window: int = 14
    atr_mult: float = 2.5
    max_hold: int = 10
    refractory_bars: int = 5
    spike_thresh: float = 0.04


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779152397"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        rank_chain = params.dd_window + params.rank_window + params.arm_window
        return int(max(params.ma_window, rank_chain,
                       params.atr_window + 1,
                       params.refractory_bars + 1)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)
        ind["close"] = close

        ind["sma"] = close.rolling(params.ma_window).mean()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        roll_max = close.rolling(params.dd_window).max()
        dd = close / roll_max - 1.0
        ind["dd"] = dd

        dd_rank = dd.rolling(params.rank_window).rank(pct=True)
        ind["dd_rank"] = dd_rank
        ind["dd_rank_min"] = dd_rank.rolling(params.arm_window).min()

        ret = close.pct_change()
        ind["spike_max"] = ret.abs().rolling(params.refractory_bars).max()

        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = indicators["close"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        dd_rank = indicators["dd_rank"].to_numpy(dtype=float)
        dd_rank_min = indicators["dd_rank_min"].to_numpy(dtype=float)
        spike_max = indicators["spike_max"].to_numpy(dtype=float)

        dd_rank_prev = np.empty(n, dtype=float)
        if n > 0:
            dd_rank_prev[0] = np.nan
            dd_rank_prev[1:] = dd_rank[:-1]

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_bar = 0
        stop_level = 0.0

        for i in range(n):
            if not in_pos:
                valid = (not np.isnan(sma[i])) and (not np.isnan(atr[i])) \
                    and (not np.isnan(dd_rank[i])) and (not np.isnan(dd_rank_prev[i])) \
                    and (not np.isnan(dd_rank_min[i])) and (not np.isnan(spike_max[i]))
                if not valid:
                    signal[i] = 0
                    continue
                regime = close[i] > sma[i]
                armed = dd_rank_min[i] < params.arm_lo
                cross = (dd_rank[i] > params.trigger_hi) and \
                        (dd_rank_prev[i] <= params.trigger_hi)
                calm = not (spike_max[i] > params.spike_thresh)
                if regime and armed and cross and calm and atr[i] > 0.0:
                    in_pos = True
                    entry_bar = i
                    stop_level = close[i] - params.atr_mult * atr[i]
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                bars_held = i - entry_bar
                exit_now = (close[i] < stop_level) or (bars_held >= params.max_hold)
                if exit_now:
                    signal[i] = 0
                    in_pos = False
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
