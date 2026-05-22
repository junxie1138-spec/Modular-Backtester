from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 200
    ac_window: int = 30
    entry_streak: int = 4
    exit_streak: int = 2
    ac_deadband: float = 0.03
    atr_window: int = 14
    vol_target: float = 0.012
    max_size: float = 1.0
    min_size: float = 0.3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778896321"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ma_window, params.ac_window + 1, params.atr_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        ind = pd.DataFrame(index=data.index)

        # 200-day MA regime gate (the hard twist)
        ind["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        ret = close.pct_change()
        ind["ret"] = ret

        # consecutive-streak counts (primary primitive)
        up = close > close.shift(1)
        down = close < close.shift(1)
        up_int = up.astype(int)
        down_int = down.astype(int)
        up_streak = up_int.groupby((~up).cumsum()).cumsum()
        down_streak = down_int.groupby((~down).cumsum()).cumsum()
        ind["up_streak"] = up_streak.astype(float)
        ind["down_streak"] = down_streak.astype(float)

        # rolling lag-1 return autocorrelation -> micro-regime classifier
        ac = ret.rolling(params.ac_window, min_periods=params.ac_window).corr(ret.shift(1))
        ind["ac"] = ac

        # ATR (as fraction of price) for volatility-scaled sizing
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ind["atr_pct"] = atr / close.replace(0.0, np.nan)

        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        up_streak = indicators["up_streak"].to_numpy(dtype=float)
        down_streak = indicators["down_streak"].to_numpy(dtype=float)
        ac = indicators["ac"].to_numpy(dtype=float)
        atr_pct = indicators["atr_pct"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        entry_k = int(params.entry_streak)
        exit_k = int(params.exit_streak)
        deadband = float(params.ac_deadband)
        min_size = float(params.min_size)
        max_size = float(params.max_size)
        vol_target = float(params.vol_target)

        in_pos = False
        mode = 0  # 1 = momentum-streak entry, -1 = reversion-streak entry
        cur_size = min_size

        for i in range(n):
            ma_i = ma[i]
            ac_i = ac[i]
            c_i = close[i]
            us = up_streak[i]
            ds = down_streak[i]

            # NaN-safe: stay flat until every gating indicator is valid
            if np.isnan(ma_i) or np.isnan(ac_i) or np.isnan(c_i) or np.isnan(us) or np.isnan(ds):
                in_pos = False
                mode = 0
                signal[i] = 0
                size[i] = 1.0
                continue

            bull = c_i > ma_i  # 200-MA regime gate

            if not in_pos:
                if bull:
                    if ac_i > deadband:
                        # momentum regime: ride a confirmed up-close streak
                        if us >= entry_k:
                            in_pos = True
                            mode = 1
                    elif ac_i < -deadband:
                        # reversion regime: buy an exhausted down-close streak
                        if ds >= entry_k:
                            in_pos = True
                            mode = -1

                if in_pos:
                    ap = atr_pct[i]
                    if np.isnan(ap) or ap <= 0.0:
                        cur_size = min_size
                    else:
                        cur_size = min(max(vol_target / ap, min_size), max_size)
                    signal[i] = 1
                    size[i] = cur_size
                else:
                    signal[i] = 0
                    size[i] = 1.0
            else:
                # signal-reversal exit: exit only when the entry condition flips
                exit_now = False
                if mode == 1:
                    # entered on an up-streak; the trigger flips when a down-streak forms
                    if ds >= exit_k:
                        exit_now = True
                else:
                    # entered on a down-streak; the trigger flips when an up-streak forms
                    if us >= exit_k:
                        exit_now = True
                # losing the 200-MA bull regime also closes the position
                if not bull:
                    exit_now = True

                if exit_now:
                    in_pos = False
                    mode = 0
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = cur_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float).clip(lower=min_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
