from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    vol_window: int = 5
    vol_ma_window: int = 5
    hold_bars: int = 6
    weekday_long: int = 1
    weekday_short: int = 4
    allow_short_leg: bool = True


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779884361"

    @classmethod
    def params_type(cls):
        return GenA1Params

    @staticmethod
    def warmup_bars(params: GenA1Params) -> int:
        vw = max(int(params.vol_window), 2)
        mw = max(int(params.vol_ma_window), 2)
        # rolling std of returns needs vw bars of returns (+1 for pct_change),
        # then a rolling mean of that std needs mw more bars.
        wb = vw + mw
        if wb > 10:
            wb = 10
        return int(wb)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        vw = max(int(params.vol_window), 2)
        mw = max(int(params.vol_ma_window), 2)

        close = data["close"].astype(float)
        ret = close.pct_change()
        vol = ret.rolling(vw, min_periods=vw).std()
        vol_ma = vol.rolling(mw, min_periods=mw).mean()

        # Compartment classification: 1.0 = Infected (vol above its short MA), 0.0 = Susceptible.
        # NaN-safe: any NaN comparison evaluates False, so warmup bars stay Susceptible.
        infected_bool = vol.gt(vol_ma)
        infected = infected_bool.astype(float)
        # Mask warmup region back to NaN so transitions cannot fire before both series are valid.
        valid = vol.notna() & vol_ma.notna()
        infected = infected.where(valid, np.nan)

        prev_infected = infected.shift(1)
        s_to_i = ((prev_infected == 0.0) & (infected == 1.0)).astype(float)
        i_to_s = ((prev_infected == 1.0) & (infected == 0.0)).astype(float)

        if isinstance(data.index, pd.DatetimeIndex):
            dow_arr = data.index.dayofweek.to_numpy().astype(float)
        else:
            dow_arr = np.full(len(data), np.nan, dtype=float)
        dow = pd.Series(dow_arr, index=data.index)

        out = pd.DataFrame(
            {
                "vol": vol,
                "vol_ma": vol_ma,
                "infected": infected,
                "s_to_i": s_to_i.fillna(0.0),
                "i_to_s": i_to_s.fillna(0.0),
                "dow": dow,
            },
            index=data.index,
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        sig = np.zeros(n, dtype=np.int64)

        dow = indicators["dow"].to_numpy()
        s_to_i = indicators["s_to_i"].fillna(0.0).to_numpy()
        i_to_s = indicators["i_to_s"].fillna(0.0).to_numpy()

        hold = int(params.hold_bars)
        if hold < 1:
            hold = 1

        weekday_long = int(params.weekday_long) % 7
        weekday_short = int(params.weekday_short) % 7
        allow_short = bool(params.allow_short_leg)

        in_pos_until = -1  # exclusive end index of current holding window
        cur_dir = 0

        for i in range(n):
            if i < in_pos_until:
                sig[i] = cur_dir
                continue

            # Flat: look for a new entry on this bar.
            d_val = dow[i]
            if d_val != d_val:  # NaN guard
                sig[i] = 0
                continue
            d = int(d_val)

            if d == weekday_long and s_to_i[i] == 1.0:
                cur_dir = 1
                in_pos_until = i + hold
                sig[i] = 1
            elif allow_short and d == weekday_short and i_to_s[i] == 1.0:
                cur_dir = -1
                in_pos_until = i + hold
                sig[i] = -1
            else:
                sig[i] = 0

        df = pd.DataFrame({"signal": sig}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
