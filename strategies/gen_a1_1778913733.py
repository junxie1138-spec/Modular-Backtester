from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_window: int = 14
    streak_threshold: int = 4
    ma_window: int = 200
    vol_window: int = 20
    target_vol: float = 0.15
    max_size: float = 1.0
    min_size: float = 0.20
    profit_target: float = 0.04
    time_stop: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778913733"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # 200-day MA dominates; pad for pct_change + rolling vol and ATR diffs.
        return int(params.ma_window + params.vol_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        atr_prev = atr.shift(1)
        # NaN comparisons yield False during warmup -> streak stays 0.
        decline = (atr < atr_prev).astype(int)
        rise = atr > atr_prev

        # Consecutive ATR-decline streak ending at each bar (vectorised).
        grp = (decline == 0).cumsum()
        decline_streak = decline.groupby(grp).cumsum()

        ma200 = close.rolling(params.ma_window).mean()

        ret = close.pct_change()
        realized_vol = ret.rolling(params.vol_window).std() * np.sqrt(252.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["decline_streak"] = decline_streak.astype(float)
        out["atr_rise"] = rise
        out["ma200"] = ma200
        out["realized_vol"] = realized_vol
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        idx = data.index

        # Compression must have built up as of the prior bar; release is today.
        prev_streak = indicators["decline_streak"].shift(1)
        coil_ready = (prev_streak >= float(params.streak_threshold)).fillna(False)
        release = indicators["atr_rise"].fillna(False)
        up_close = (close > close.shift(1)).fillna(False)
        regime = (close > indicators["ma200"]).fillna(False)

        entry = (coil_ready & release & up_close & regime).to_numpy()
        close_arr = close.to_numpy(dtype=float)
        n = len(close_arr)

        # Path-dependent exit: profit-target OR time-stop, whichever first.
        sig = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(n):
            if in_pos:
                held = i - entry_bar
                gain = close_arr[i] / entry_price - 1.0
                if gain >= params.profit_target or held >= params.time_stop:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    entry_price = close_arr[i]
                    entry_bar = i
                    sig[i] = 1

        # Volatility-targeting: size inversely scaled to realized vol.
        rv = indicators["realized_vol"].replace(0.0, np.nan)
        size = (params.target_vol / rv).clip(
            lower=params.min_size, upper=params.max_size
        )
        size = size.fillna(params.min_size)

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(sig, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = size.shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
