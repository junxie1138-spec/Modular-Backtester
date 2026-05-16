from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    prey_window: int = 40
    predator_window: int = 10
    rank_window: int = 252
    entry_pct_hi: int = 85
    entry_pct_lo: int = 15
    vol_window: int = 20
    atr_window: int = 14
    breakeven_pct: float = 1.5
    trail_k: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778904695"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + params.prey_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        cc_ret = data["close"].pct_change()

        prey = cc_ret.rolling(params.prey_window, min_periods=params.prey_window).sum()
        bear_press = (-cc_ret).clip(lower=0)
        predator = bear_press.rolling(params.predator_window, min_periods=params.predator_window).sum()

        # Predation ratio: undefined when prey (bull return) is absent
        ind["predation_ratio"] = np.where(prey > 0, predator / (prey + 1e-8), np.nan)

        # Rolling percentile rank — the hard twist
        ind["ratio_rank"] = (
            ind["predation_ratio"]
            .rolling(params.rank_window, min_periods=params.rank_window)
            .rank(pct=True)
        )

        ind["realized_vol"] = cc_ret.rolling(params.vol_window, min_periods=params.vol_window).std()

        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        ratio_rank = indicators["ratio_rank"].values
        realized_vol = indicators["realized_vol"].values
        atr = indicators["atr"].values

        n = len(data)
        raw_signal = np.zeros(n, dtype=int)

        # Vol-targeted sizing: constant target daily vol exposure
        target_vol = 0.01
        size_arr = np.where(
            realized_vol > 1e-8,
            np.clip(target_vol / realized_vol, 0.5, 2.0),
            1.0,
        )

        hi_thresh = params.entry_pct_hi / 100.0
        lo_thresh = params.entry_pct_lo / 100.0
        be_mult_long = 1.0 + params.breakeven_pct / 100.0
        be_mult_short = 1.0 - params.breakeven_pct / 100.0
        trail_k = params.trail_k

        # Path-dependent breakeven-trail exit state
        in_trade = False
        direction = 0
        entry_price = 0.0
        stop = 0.0
        be_reached = False
        extreme = 0.0  # peak (longs) or trough (shorts) since entry

        for i in range(n):
            c = close[i]
            a = atr[i]
            pr = ratio_rank[i]

            if in_trade:
                if np.isnan(a):
                    raw_signal[i] = direction
                    continue

                if direction == 1:
                    if c > extreme:
                        extreme = c
                    if not be_reached and c >= entry_price * be_mult_long:
                        stop = entry_price
                        be_reached = True
                    if be_reached:
                        new_stop = extreme - trail_k * a
                        if new_stop > stop:  # stop only ever moves up
                            stop = new_stop
                    if c <= stop:
                        raw_signal[i] = 0
                        in_trade = False
                        direction = 0
                    else:
                        raw_signal[i] = 1

                else:  # direction == -1
                    if c < extreme:
                        extreme = c
                    if not be_reached and c <= entry_price * be_mult_short:
                        stop = entry_price
                        be_reached = True
                    if be_reached:
                        new_stop = extreme + trail_k * a
                        if new_stop < stop:  # stop only ever moves down
                            stop = new_stop
                    if c >= stop:
                        raw_signal[i] = 0
                        in_trade = False
                        direction = 0
                    else:
                        raw_signal[i] = -1

            else:
                if np.isnan(pr) or np.isnan(a):
                    continue

                if pr >= hi_thresh:
                    # Extreme predation -> expect prey recovery -> long
                    in_trade = True
                    direction = 1
                    entry_price = c
                    extreme = c
                    stop = c - trail_k * a
                    be_reached = False
                    raw_signal[i] = 1

                elif pr <= lo_thresh:
                    # Prey dominance -> predators return -> short
                    in_trade = True
                    direction = -1
                    entry_price = c
                    extreme = c
                    stop = c + trail_k * a
                    be_reached = False
                    raw_signal[i] = -1

        # Mandatory 1-bar shift: decision at bar N fills at bar N+1
        signal_series = pd.Series(raw_signal, index=data.index)
        signal_series = signal_series.shift(1).fillna(0).astype(int)

        df = pd.DataFrame({"signal": signal_series, "size": size_arr}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
