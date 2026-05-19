from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapHoldParams:
    min_gap: float = 0.002
    atr_len: int = 14
    sma_len: int = 100
    range_mult: float = 1.5
    be_pct: float = 0.03
    k_atr: float = 2.5
    max_hold: int = 12
    size_ref: float = 0.6
    size_floor: float = 0.3
    size_cap: float = 1.0


class GeneratedStrategy(BaseStrategy[GapHoldParams]):
    strategy_id = "gen_a2_1779149989"

    @classmethod
    def params_type(cls):
        return GapHoldParams

    @staticmethod
    def warmup_bars(params: GapHoldParams) -> int:
        return int(max(params.atr_len, params.sma_len)) + 1

    def indicators(self, data: pd.DataFrame, params: GapHoldParams) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prior_close = close.shift(1)
        gap = open_ / prior_close - 1.0

        tr1 = high - low
        tr2 = (high - prior_close).abs()
        tr3 = (low - prior_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()
        sma = close.rolling(params.sma_len, min_periods=params.sma_len).mean()

        gap_up = gap > params.min_gap
        held = low >= prior_close
        contained = tr <= params.range_mult * atr
        regime = close > sma
        entry_cond = (gap_up & held & contained & regime).fillna(False)

        safe_atr = atr.replace(0.0, np.nan)
        hold_strength = (low - prior_close) / safe_atr

        out = pd.DataFrame(index=data.index)
        out["prior_close"] = prior_close
        out["atr"] = atr
        out["entry_cond"] = entry_cond.astype(bool)
        out["hold_strength"] = hold_strength
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapHoldParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_cond = indicators["entry_cond"].to_numpy()
        hold_strength = indicators["hold_strength"].to_numpy(dtype=float)

        n = len(close)
        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        position = 0
        entry_price = 0.0
        stop = 0.0
        be_armed = False
        bars_held = 0
        cur_size = 1.0

        for i in range(n):
            atr_i = atr[i]
            atr_ok = np.isfinite(atr_i) and atr_i > 0.0

            if position == 1:
                bars_held += 1
                if atr_ok:
                    if high[i] >= entry_price * (1.0 + params.be_pct):
                        be_armed = True
                        if entry_price > stop:
                            stop = entry_price
                    if be_armed:
                        trail = high[i] - params.k_atr * atr_i
                        if trail > stop:
                            stop = trail
                if low[i] <= stop:
                    position = 0
                    sig[i] = 0
                elif bars_held >= params.max_hold:
                    position = 0
                    sig[i] = 0
                else:
                    sig[i] = 1
                    size[i] = cur_size
                continue

            if atr_ok and bool(entry_cond[i]):
                position = 1
                entry_price = close[i]
                stop = close[i] - params.k_atr * atr_i
                be_armed = False
                bars_held = 0
                hs = hold_strength[i]
                if not np.isfinite(hs):
                    hs = 0.0
                cur_size = float(
                    np.clip(hs / params.size_ref, params.size_floor, params.size_cap)
                )
                sig[i] = 1
                size[i] = cur_size
            else:
                sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
