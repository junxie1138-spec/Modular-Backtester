from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TurnOfMonthVolParams:
    atr_len: int = 14
    sma_len: int = 200
    vol_lookback: int = 120
    vol_pct: float = 0.5
    tom_pre: int = 3
    tom_post: int = 2
    k_stop: float = 2.0
    hold_bars: int = 2
    capacity: int = 1


class GeneratedStrategy(BaseStrategy[TurnOfMonthVolParams]):
    strategy_id = "gen_a1_1778910803"

    @classmethod
    def params_type(cls):
        return TurnOfMonthVolParams

    @staticmethod
    def warmup_bars(params: TurnOfMonthVolParams) -> int:
        return int(max(params.sma_len, params.vol_lookback + params.atr_len + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: TurnOfMonthVolParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        sma = close.rolling(params.sma_len, min_periods=params.sma_len).mean()

        atr_pct = atr.rolling(
            params.vol_lookback, min_periods=params.vol_lookback
        ).rank(pct=True)

        idx = data.index
        months = pd.Series(
            np.asarray(idx.year) * 12 + np.asarray(idx.month), index=idx
        )
        order = pd.Series(np.arange(len(data)), index=idx)
        grp = order.groupby(months)
        pos_from_start = grp.cumcount()
        pos_from_end = grp.cumcount(ascending=False)
        in_tom = (
            (pos_from_end < int(params.tom_pre))
            | (pos_from_start < int(params.tom_post))
        ).astype(float)

        out = pd.DataFrame(index=idx)
        out["atr"] = atr
        out["sma"] = sma
        out["atr_pct"] = atr_pct
        out["in_tom"] = in_tom
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TurnOfMonthVolParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        atr_pct = indicators["atr_pct"].to_numpy(dtype=float)
        in_tom = indicators["in_tom"].to_numpy(dtype=float) > 0.5

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        entries_in_window = 0
        cap = max(1, int(params.capacity))
        hold = max(1, int(params.hold_bars))

        for i in range(n):
            if not in_tom[i]:
                entries_in_window = 0

            if in_pos:
                bars_held += 1
                stop_level = entry_price - params.k_stop * entry_atr
                if close[i] <= stop_level or bars_held >= hold:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
                continue

            valid = (
                not np.isnan(sma[i])
                and not np.isnan(atr[i])
                and not np.isnan(atr_pct[i])
                and atr[i] > 0.0
            )
            if (
                valid
                and in_tom[i]
                and close[i] > sma[i]
                and atr_pct[i] <= params.vol_pct
                and entries_in_window < cap
            ):
                in_pos = True
                entry_price = close[i]
                entry_atr = atr[i]
                bars_held = 0
                entries_in_window += 1
                signal[i] = 1
            else:
                signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
