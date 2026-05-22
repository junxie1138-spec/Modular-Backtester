from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_window: int = 3
    rank_window: int = 60
    pos_window: int = 20
    entry_q: float = 0.15
    pos_q: float = 0.30
    atr_window: int = 14
    k_atr: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779145396"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(
            max(
                params.shock_window + params.rank_window,
                params.pos_window + params.rank_window,
                params.atr_window,
            )
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # close-to-close returns and the cumulative 'shockwave' drift
        ret = close.pct_change()
        shock = ret.rolling(params.shock_window).sum()
        # percentile threshold: low quantile of the shockwave's own distribution
        shock_thr = shock.rolling(params.rank_window).quantile(params.entry_q)

        # relative position of close within its recent range, in [0, 1]
        roll_lo = close.rolling(params.pos_window).min()
        roll_hi = close.rolling(params.pos_window).max()
        rng = roll_hi - roll_lo
        pos = (close - roll_lo) / rng.where(rng > 0)
        # percentile threshold on the relative position
        pos_thr = pos.rolling(params.rank_window).quantile(params.pos_q)

        # ATR for the fixed volatility stop
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

        out = pd.DataFrame(index=data.index)
        out["shock"] = shock
        out["shock_thr"] = shock_thr
        out["pos"] = pos
        out["pos_thr"] = pos_thr
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        idx = data.index
        close_arr = data["close"].to_numpy(dtype=float)
        atr_arr = indicators["atr"].to_numpy(dtype=float)

        shock = indicators["shock"]
        shock_thr = indicators["shock_thr"]
        pos = indicators["pos"]
        pos_thr = indicators["pos_thr"]

        # entry: shockwave in low percentile AND relative position in low percentile
        raw_entry = (
            ((shock <= shock_thr) & (pos <= pos_thr))
            .fillna(False)
            .to_numpy(dtype=bool)
        )

        n = len(idx)
        signal = np.zeros(n, dtype=int)
        in_pos = False
        stop_level = 0.0
        bars_held = 0
        max_hold = max(1, int(params.max_hold))

        for i in range(n):
            if in_pos:
                bars_held += 1
                c = close_arr[i]
                hit_stop = (not np.isnan(c)) and (c <= stop_level)
                if hit_stop or bars_held >= max_hold:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            if not in_pos and raw_entry[i]:
                a = atr_arr[i]
                c = close_arr[i]
                if np.isfinite(a) and a > 0.0 and np.isfinite(c):
                    in_pos = True
                    # fixed volatility stop anchored at entry price and entry-bar ATR
                    stop_level = c - params.k_atr * a
                    bars_held = 0
                    signal[i] = 1

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(signal, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
