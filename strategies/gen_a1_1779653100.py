from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    weekday_lookback: int = 50
    atr_window: int = 14
    atr_stop_mult: float = 2.5
    upper_pct: float = 0.80
    lower_pct: float = 0.20
    max_hold_bars: int = 10
    min_warmup_bars: int = 260
    use_short_side: bool = True


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779653100"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        wd_needed = params.weekday_lookback * 5 + 5
        return int(max(wd_needed, params.atr_window + 1, params.min_warmup_bars))

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        weekday = pd.Series(data.index.dayofweek, index=data.index)

        pct_rank = pd.Series(np.nan, index=data.index, dtype=float)
        min_p = max(5, params.weekday_lookback // 2)
        for d in range(5):
            mask = weekday == d
            if not mask.any():
                continue
            sub = ret[mask]
            if len(sub) == 0:
                continue
            sub_rank = sub.rolling(
                window=params.weekday_lookback,
                min_periods=min_p,
            ).rank(pct=True)
            pct_rank.loc[sub_rank.index] = sub_rank

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=params.atr_window, min_periods=params.atr_window).mean()

        out = pd.DataFrame(
            {
                "ret": ret,
                "weekday": weekday.astype(float),
                "wd_rank": pct_rank,
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
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        wd_rank = indicators["wd_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        finite_rank = np.isfinite(wd_rank)
        upper_hit = finite_rank & (wd_rank >= params.upper_pct)
        lower_hit = finite_rank & (wd_rank <= params.lower_pct)

        long_trigger = np.zeros(n, dtype=bool)
        short_trigger = np.zeros(n, dtype=bool)
        if n >= 2:
            long_trigger[1:] = lower_hit[1:] & lower_hit[:-1]
            short_trigger[1:] = upper_hit[1:] & upper_hit[:-1]

        raw_signal = np.zeros(n, dtype=np.int64)

        position = 0
        entry_idx = -1
        hwm = -np.inf
        lwm = np.inf
        entry_atr = np.nan

        for i in range(n):
            c = close[i]
            a = atr[i]

            if position == 1:
                if c > hwm:
                    hwm = c
                stop_level = hwm - params.atr_stop_mult * entry_atr
                bars_held = i - entry_idx
                if (not np.isfinite(c)) or c <= stop_level or bars_held >= params.max_hold_bars:
                    position = 0
                    entry_idx = -1
                    hwm = -np.inf
                    entry_atr = np.nan
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            elif position == -1:
                if c < lwm:
                    lwm = c
                stop_level = lwm + params.atr_stop_mult * entry_atr
                bars_held = i - entry_idx
                if (not np.isfinite(c)) or c >= stop_level or bars_held >= params.max_hold_bars:
                    position = 0
                    entry_idx = -1
                    lwm = np.inf
                    entry_atr = np.nan
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = -1
            else:
                if (not np.isfinite(a)) or a <= 0 or (not np.isfinite(c)):
                    raw_signal[i] = 0
                    continue
                if long_trigger[i]:
                    position = 1
                    entry_idx = i
                    hwm = c
                    entry_atr = a
                    raw_signal[i] = 1
                elif params.use_short_side and short_trigger[i]:
                    position = -1
                    entry_idx = i
                    lwm = c
                    entry_atr = a
                    raw_signal[i] = -1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
