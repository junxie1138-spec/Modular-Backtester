from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_len: int = 20
    atr_len: int = 14
    dislocation_k: float = 1.5
    vol_short_len: int = 5
    vol_long_len: int = 30
    vol_ratio_max: float = 0.9
    trail_k: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778909535"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.ma_len, params.atr_len + 1, params.vol_long_len + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)

        ma = close.rolling(params.ma_len).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        # Primitive A: dislocation depth in ATR units (price stretched below MA).
        safe_atr = atr.replace(0.0, np.nan)
        dislocation = (ma - close) / safe_atr

        # Primitive B: realized-volatility contraction regime.
        rets = close.pct_change()
        vol_short = rets.rolling(params.vol_short_len).std()
        vol_long = rets.rolling(params.vol_long_len).std()
        safe_vol_long = vol_long.replace(0.0, np.nan)
        vol_ratio = vol_short / safe_vol_long

        ind["atr"] = atr
        ind["dislocation"] = dislocation
        ind["vol_ratio"] = vol_ratio
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        dislocation = indicators["dislocation"].to_numpy(dtype=float)
        vol_ratio = indicators["vol_ratio"].to_numpy(dtype=float)

        n = len(data)
        signal = np.zeros(n, dtype=int)

        # Two-primitive AND: deep ATR-dislocation AND contracting vol regime.
        entry_ok = (
            np.isfinite(dislocation)
            & np.isfinite(vol_ratio)
            & np.isfinite(atr)
            & (dislocation >= params.dislocation_k)
            & (vol_ratio <= params.vol_ratio_max)
        )

        position = 0
        hwm = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                if entry_ok[i]:
                    position = 1
                    hwm = close[i]
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.trail_k * atr[i]
                exit_now = bars_held >= params.max_hold
                if np.isfinite(stop) and close[i] < stop:
                    exit_now = True
                if exit_now:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
