from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapOrdinalParams:
    lookback: int = 63
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapOrdinalParams]):
    strategy_id = "gen_local-w2_1791223453"

    @classmethod
    def params_type(cls) -> type[GapOrdinalParams]:
        return GapOrdinalParams

    @classmethod
    def warmup_bars(cls, params: GapOrdinalParams) -> int:
        return params.lookback + 2

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapOrdinalParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        raw_gap = data["open"].div(previous_close).sub(1.0)
        gap = raw_gap.fillna(0.0)

        a = raw_gap.shift(2)
        b = raw_gap.shift(1)
        c = raw_gap
        valid = a.notna() & b.notna() & c.notna()

        pattern = pd.Series(5, index=data.index, dtype=int)
        pattern = pattern.mask((a <= b) & (b <= c), 0)
        pattern = pattern.mask((a <= c) & (c < b), 1)
        pattern = pattern.mask((b < a) & (a <= c), 2)
        pattern = pattern.mask((b <= c) & (c < a), 3)
        pattern = pattern.mask((c < a) & (a <= b), 4)

        counts = []
        for code in range(6):
            counts.append(((pattern == code) & valid).astype(float).rolling(params.lookback).sum())

        total = valid.astype(float).rolling(params.lookback).sum()
        probabilities = pd.concat(counts, axis=1).div(total.replace(0.0, np.nan), axis=0)
        entropy_terms = probabilities * np.log(probabilities.where(probabilities > 0.0))
        entropy = -entropy_terms.sum(axis=1) / np.log(6.0)
        disequilibrium = probabilities.sub(1.0 / 6.0).pow(2).sum(axis=1) / (5.0 / 6.0)
        complexity = (entropy * disequilibrium).where(total >= params.lookback, 0.0).fillna(0.0)

        gap_volatility = raw_gap.rolling(params.lookback).std(ddof=0).fillna(0.0)
        ready = (total >= params.lookback).astype(float).fillna(0.0)

        return pd.DataFrame(
            {
                "gap": gap,
                "gap_volatility": gap_volatility,
                "ordinal_complexity": complexity,
                "ready": ready,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapOrdinalParams,
    ) -> SignalFrame:
        entry = (
            (indicators["ready"] > 0.0)
            & (indicators["ordinal_complexity"] > 0.08)
            & (indicators["gap_volatility"] > 0.0)
            & (indicators["gap"] < -1.25 * indicators["gap_volatility"])
            & (data["close"] > data["open"])
            & (data["close"] >= data["close"].shift(1))
        ).fillna(False)

        signal = pd.Series(0, index=data.index, dtype=int)
        remaining = 0
        exited_last_bar = False
        for i in range(len(data)):
            if remaining > 0:
                signal.iat[i] = 1
                remaining -= 1
                exited_last_bar = remaining == 0
            elif exited_last_bar:
                exited_last_bar = False
            elif bool(entry.iat[i]):
                signal.iat[i] = 1
                remaining = params.hold_bars - 1
                exited_last_bar = remaining == 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
