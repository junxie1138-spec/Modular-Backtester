from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    autocorr_window: int = 42
    rank_window: int = 252
    entry_percentile: float = 0.8
    momentum_window: int = 15
    atr_window: int = 14
    breakeven_pct: float = 0.04
    trail_atr: float = 3.0
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713894"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return params.autocorr_window + params.rank_window + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        autocorr = ret.rolling(params.autocorr_window).corr(ret.shift(1))
        rank = autocorr.rolling(params.rank_window).rank(pct=True)
        momentum = data["close"].pct_change(params.momentum_window)
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        return pd.DataFrame({"autocorr": autocorr, "rank": rank, "momentum": momentum, "atr": atr}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entry = (
            (indicators["rank"] >= params.entry_percentile)
            & (indicators["autocorr"] > 0)
            & (indicators["momentum"] > 0)
            & indicators[["rank", "autocorr", "momentum", "atr"]].notna().all(axis=1)
        )
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        peak = 0.0
        stop = 0.0
        held = 0
        for i in range(len(data)):
            if active:
                held += 1
                peak = max(peak, float(data["high"].iloc[i]))
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    stop = max(stop, entry_price, peak - params.trail_atr * float(indicators["atr"].iloc[i]))
                if float(data["low"].iloc[i]) <= stop or held >= params.max_hold_bars:
                    active = False
                    raw[i] = 0
                    continue
                raw[i] = 1
            elif bool(entry.iloc[i]):
                active = True
                entry_price = float(data["close"].iloc[i])
                peak = entry_price
                stop = 0.0
                held = 0
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
