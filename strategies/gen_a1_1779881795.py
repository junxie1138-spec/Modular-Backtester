from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEntrainmentParams:
    gap_window: int = 20
    rank_window: int = 252
    freq_threshold: float = 0.75
    mag_threshold: float = 0.75
    atr_window: int = 14
    atr_stop_mult: float = 3.0
    min_gap_eps: float = 0.0


class GeneratedStrategy(BaseStrategy[GapEntrainmentParams]):
    strategy_id = "gen_a1_1779881795"

    @classmethod
    def params_type(cls):
        return GapEntrainmentParams

    @classmethod
    def warmup_bars(cls, params: GapEntrainmentParams) -> int:
        return int(params.rank_window + params.gap_window + params.atr_window + 5)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapEntrainmentParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        gap_ret = (data["open"] - prev_close) / prev_close

        is_up_gap = (gap_ret > params.min_gap_eps).astype(float)
        # If gap_ret is NaN (first bar), is_up_gap is 0 from the comparison; mask it back to NaN
        is_up_gap = is_up_gap.where(gap_ret.notna(), np.nan)

        gw = max(int(params.gap_window), 2)
        rw = max(int(params.rank_window), gw + 1)
        aw = max(int(params.atr_window), 2)

        gap_freq = is_up_gap.rolling(gw, min_periods=gw).mean()
        gap_mag = gap_ret.rolling(gw, min_periods=gw).sum()

        freq_rank = gap_freq.rolling(rw, min_periods=rw).rank(pct=True)
        mag_rank = gap_mag.rolling(rw, min_periods=rw).rank(pct=True)

        prev_close_atr = data["close"].shift(1)
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close_atr).abs()
        tr3 = (data["low"] - prev_close_atr).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(aw, min_periods=aw).mean()

        out = pd.DataFrame(
            {
                "gap_ret": gap_ret,
                "gap_freq": gap_freq,
                "gap_mag": gap_mag,
                "freq_rank": freq_rank,
                "mag_rank": mag_rank,
                "atr": atr,
            },
            index=data.index,
        )
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEntrainmentParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(copy=False)
        atr = indicators["atr"].to_numpy(copy=False)
        freq_rank = indicators["freq_rank"].to_numpy(copy=False)
        mag_rank = indicators["mag_rank"].to_numpy(copy=False)
        n = len(close)

        freq_thr = float(params.freq_threshold)
        mag_thr = float(params.mag_threshold)
        k = float(params.atr_stop_mult)

        raw_sig = np.zeros(n, dtype=np.int64)
        in_pos = False
        hwm = -np.inf

        for i in range(n):
            c = close[i]
            a = atr[i]
            fr = freq_rank[i]
            mr = mag_rank[i]

            if np.isnan(c) or np.isnan(a):
                if in_pos:
                    # Cannot evaluate stop without ATR; stay flat to be safe
                    in_pos = False
                    hwm = -np.inf
                continue

            if not in_pos:
                if (
                    not np.isnan(fr)
                    and not np.isnan(mr)
                    and fr >= freq_thr
                    and mr >= mag_thr
                ):
                    in_pos = True
                    hwm = c
                    raw_sig[i] = 1
            else:
                if c > hwm:
                    hwm = c
                stop_level = hwm - k * a
                if c < stop_level:
                    raw_sig[i] = 0
                    in_pos = False
                    hwm = -np.inf
                else:
                    raw_sig[i] = 1

        size = np.ones(n, dtype=np.float64)

        df = pd.DataFrame(
            {
                "signal": raw_sig,
                "size": size,
            },
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
