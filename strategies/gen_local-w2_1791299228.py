from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    lookback: int = 80
    shock_pct: float = 0.15
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = "gen_local-w2_1791299228"

    @classmethod
    def params_type(cls):
        return RangeMigrationParams

    def warmup_bars(self, params: RangeMigrationParams) -> int:
        return params.lookback + 25

    def indicators(self, data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        width = (data["high"] - data["low"]).clip(lower=0)
        center = (data["high"] + data["low"]) / 2
        out["center"] = center
        out["migration"] = ((center - center.shift(1)) / width.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        out["shock_cut"] = out["migration"].rolling(params.lookback, min_periods=params.lookback).quantile(params.shock_pct).shift(1).fillna(-np.inf)
        out["drawdown"] = (data["close"] / data["close"].rolling(20, min_periods=20).max() - 1).fillna(0)
        out["drawdown_cut"] = out["drawdown"].rolling(params.lookback, min_periods=params.lookback).quantile(0.30).shift(1).fillna(-np.inf)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([width, (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean().fillna(0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeMigrationParams,
    ) -> SignalFrame:
        entry = (
            (indicators["migration"].shift(1) < indicators["shock_cut"].shift(1))
            & (indicators["drawdown"].shift(1) < indicators["drawdown_cut"].shift(1))
            & (indicators["migration"] > 0)
            & (data["low"] > data["low"].shift(1))
            & (data["close"] > indicators["center"].shift(1))
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry_price = peak = stop = 0.0
        held = 0

        for i in range(len(data)):
            if pending:
                active = True
                pending = False
                entry_price = peak = opens[i]
                stop = -np.inf
                armed = False
                held = 0

            exited = False
            if active:
                held += 1
                if (armed and lows[i] <= stop) or held >= params.hold_bars:
                    active = False
                    exited = True
                else:
                    if highs[i] >= entry_price * (1 + params.breakeven_pct):
                        armed = True
                    if armed:
                        peak = max(peak, highs[i])
                        stop = max(stop, entry_price, peak - params.trail_atr * atr[i])

            if active:
                signal[i] = 1
            elif not exited and i >= ctx.warmup_bars and i + 1 < len(data) and entry[i]:
                pending = True
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
