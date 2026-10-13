from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_pct: float = 0.004
    volume_mult: float = 1.2
    trail_atr: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713708"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return 22

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        vol_avg = data["volume"].rolling(20, min_periods=20).mean()
        out = pd.DataFrame(index=data.index)
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        out["vol_avg"] = vol_avg
        out["gap_up"] = (data["open"] / prev_close - 1.0) >= params.gap_pct
        out["setup"] = out["gap_up"] & (data["volume"] >= params.volume_mult * vol_avg)
        out["confirm"] = (data["close"] > prev_close) & (data["volume"] >= params.volume_mult * vol_avg)
        out["entry"] = out["setup"].shift(2, fill_value=False) & out["confirm"].shift(1, fill_value=False) & out["confirm"]
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        high_water = 0.0
        held = 0
        for i, close in enumerate(closes):
            if active:
                held += 1
                high_water = max(high_water, close)
                stop = high_water - params.trail_atr * atr[i] if np.isfinite(atr[i]) else -np.inf
                if close <= stop or held >= params.max_hold:
                    active = False
                    continue
                raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                high_water = close
                held = 0
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
