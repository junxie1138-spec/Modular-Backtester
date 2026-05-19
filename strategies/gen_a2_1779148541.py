from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeekdaySeasonParams:
    vol_window: int = 20
    vol_mult: float = 1.10
    season_window: int = 26
    min_season_obs: int = 8
    min_score: float = 0.0
    profit_target: float = 0.015
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[WeekdaySeasonParams]):
    strategy_id = "gen_a2_1779148541"

    @classmethod
    def params_type(cls):
        return WeekdaySeasonParams

    @staticmethod
    def warmup_bars(params: WeekdaySeasonParams) -> int:
        # season_window is counted in per-weekday instances (~1 per 5 calendar
        # bars); convert to calendar bars and pad.
        return int(max(params.vol_window + 1, params.season_window * 5 + 5))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WeekdaySeasonParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        n = len(data)

        ret = close.pct_change()
        wd = np.asarray(data.index.dayofweek)

        # Volume-confirmed up move: an up close on above-average volume.
        avg_vol = volume.rolling(params.vol_window, min_periods=params.vol_window).mean()
        up = close > close.shift(1)
        vol_conf = ((volume > avg_vol * params.vol_mult) & up).fillna(False)
        # Two-bar confirmation: this bar AND the prior bar both confirmed.
        two_bar = vol_conf & vol_conf.shift(1).fillna(False)

        # Rolling mean return for each weekday, using only that weekday's own
        # prior instances (shift(1) drops the current instance -> no lookahead).
        scores = pd.DataFrame(index=data.index)
        for d in range(5):
            sub = ret[wd == d]
            rm = sub.rolling(params.season_window, min_periods=params.min_season_obs).mean().shift(1)
            scores[d] = rm.reindex(data.index).ffill()

        score_mat = scores.to_numpy()
        own = score_mat[np.arange(n), wd]
        own_s = pd.Series(own, index=data.index)
        best_s = scores.max(axis=1)  # NaN-safe; all-NaN warmup rows -> NaN

        # Current weekday is the seasonally top-ranked one and has a positive
        # edge, with two-bar volume confirmation present.
        is_best = own_s >= best_s
        entry_ok = (two_bar & is_best & (own_s > params.min_score)).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["entry_ok"] = entry_ok.astype(bool)
        out["own_score"] = own_s
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WeekdaySeasonParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        entry_ok = indicators["entry_ok"].to_numpy(dtype=bool)
        n = len(data)

        pos = np.zeros(n, dtype=np.int64)
        holding = False
        entry_idx = -1
        entry_price = 0.0

        for i in range(n):
            if not holding:
                if entry_ok[i]:
                    holding = True
                    entry_idx = i
                    entry_price = close[i]
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                bars_held = i - entry_idx
                gain = (close[i] / entry_price) - 1.0 if entry_price > 0.0 else 0.0
                # Profit-target OR time-stop, whichever fires first.
                if gain >= params.profit_target or bars_held >= params.max_hold:
                    pos[i] = 0
                    holding = False
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
