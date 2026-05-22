from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_len: int = 20
    smooth_len: int = 3
    rank_len: int = 60
    vol_rank_len: int = 60
    atr_len: int = 14
    entry_pct: float = 0.85
    vol_pct: float = 0.70
    trail_k: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779152741"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        base = params.channel_len + params.smooth_len + params.rank_len
        return int(max(base, params.vol_rank_len, params.atr_len + 1)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        out = pd.DataFrame(index=data.index)

        # Tide level: close-location-value inside a rolling high-low channel.
        roll_max = high.rolling(params.channel_len).max()
        roll_min = low.rolling(params.channel_len).min()
        rng = (roll_max - roll_min).replace(0.0, np.nan)
        clv = ((close - roll_min) / rng).clip(0.0, 1.0)
        # flat-range bars (rng==0) are neutral tide; warmup bars stay NaN.
        clv = clv.where(~(rng.isna() & roll_max.notna()), 0.5)

        clv_smooth = clv.rolling(params.smooth_len).mean()

        # Percentile-rank threshold (the twist): rank of the tide level itself.
        clv_rank = clv_smooth.rolling(params.rank_len).rank(pct=True)
        vol_rank = volume.rolling(params.vol_rank_len).rank(pct=True)

        # ATR for the rolling-high trailing stop.
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        atr = tr.rolling(params.atr_len).mean()

        out["clv"] = clv
        out["clv_smooth"] = clv_smooth
        out["clv_rank"] = clv_rank
        out["vol_rank"] = vol_rank
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        clv_rank = indicators["clv_rank"]
        vol_rank = indicators["vol_rank"]
        atr = indicators["atr"].to_numpy(dtype=float)

        # Rising tide: rank crosses up through the top band, volume-confirmed.
        cr = clv_rank
        cr_prev = clv_rank.shift(1)
        cross_up = (
            (cr >= params.entry_pct)
            & (cr_prev < params.entry_pct)
            & (vol_rank >= params.vol_pct)
        ).fillna(False).to_numpy()

        n = len(data)
        pos = np.zeros(n, dtype=int)
        in_pos = False
        hwm = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if cross_up[i] and not np.isnan(atr[i]):
                    in_pos = True
                    hwm = close[i]
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.trail_k * atr[i]
                if (not np.isnan(atr[i]) and close[i] < stop) or bars_held >= params.max_hold:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
