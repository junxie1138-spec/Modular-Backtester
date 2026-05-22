from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 20
    rank_lookback: int = 120
    transmission_rank_thresh: float = 0.80
    load_rank_thresh: float = 0.80
    atr_window: int = 14
    atr_mult: float = 2.5
    max_hold_bars: int = 18
    gap_up_thresh: float = 0.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779153679"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.rank_lookback + params.gap_window + 20)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap_ret = open_ / prev_close - 1.0

        # SI epidemic framing: an up-gap day is an 'infection' event.
        up_gap = (gap_ret > params.gap_up_thresh).astype(float)
        infected = up_gap.rolling(params.gap_window).mean()
        susceptible = 1.0 - infected
        # Transmission term S*I peaks mid-epidemic (active spread phase).
        transmission = susceptible * infected

        # Cumulative 'viral load' = summed positive overnight gap force.
        pos_gap = gap_ret.clip(lower=0.0)
        viral_load = pos_gap.rolling(params.gap_window).sum()

        lookback = int(params.rank_lookback)
        transmission_rank = transmission.rolling(lookback).rank(pct=True)
        load_rank = viral_load.rolling(lookback).rank(pct=True)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["transmission_rank"] = transmission_rank
        out["load_rank"] = load_rank
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        t_rank = indicators["transmission_rank"].to_numpy(dtype=float)
        l_rank = indicators["load_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                exit_now = False
                # Fixed volatility-stop: stop_level frozen at entry.
                if not np.isnan(close[i]) and close[i] <= stop_level:
                    exit_now = True
                if bars_held >= params.max_hold_bars:
                    exit_now = True
                if exit_now:
                    in_pos = False
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = 1
                continue

            tr_ok = (not np.isnan(t_rank[i])) and t_rank[i] >= params.transmission_rank_thresh
            ld_ok = (not np.isnan(l_rank[i])) and l_rank[i] >= params.load_rank_thresh
            atr_ok = (not np.isnan(atr[i])) and atr[i] > 0.0

            # Two-primitive AND: epidemic stage rank and viral-load rank agree.
            if tr_ok and ld_ok and atr_ok and not np.isnan(close[i]):
                in_pos = True
                bars_held = 0
                stop_level = close[i] - params.atr_mult * atr[i]
                raw[i] = 1
            else:
                raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
