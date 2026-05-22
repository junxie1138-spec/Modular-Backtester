from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class YieldPointParams:
    atr_len: int = 14
    excursion_window: int = 20
    ma_len: int = 200
    yield_strain: float = 4.0
    retention_thresh: float = 0.65
    recovery_max: float = 1.5
    breakeven_target: float = 0.04
    trail_k: float = 3.0
    init_stop_k: float = 2.5
    max_hold: int = 15


class GeneratedStrategy(BaseStrategy[YieldPointParams]):
    strategy_id = "gen_a1_1778893728"

    @classmethod
    def params_type(cls) -> type[YieldPointParams]:
        return YieldPointParams

    @staticmethod
    def warmup_bars(params: YieldPointParams) -> int:
        return int(max(params.ma_len, params.excursion_window, params.atr_len)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: YieldPointParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()
        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        rmax = close.rolling(params.excursion_window, min_periods=params.excursion_window).max()
        rmin = close.rolling(params.excursion_window, min_periods=params.excursion_window).min()

        atr_safe = atr.where(atr > 0)
        total_excursion = (rmax - rmin) / atr_safe
        plastic_strain = (close - rmin) / atr_safe
        elastic_recovery = (rmax - close) / atr_safe
        plastic_ratio = plastic_strain / total_excursion.where(total_excursion > 0)

        entry_cond = (
            (total_excursion >= params.yield_strain)
            & (plastic_ratio >= params.retention_thresh)
            & (elastic_recovery <= params.recovery_max)
        ).fillna(False)
        fresh = entry_cond & (~entry_cond.shift(1).fillna(False))

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["ma"] = ma
        out["total_excursion"] = total_excursion
        out["plastic_ratio"] = plastic_ratio
        out["elastic_recovery"] = elastic_recovery
        out["fresh_entry"] = fresh.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: YieldPointParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        fresh = indicators["fresh_entry"].to_numpy(dtype=float)

        n = len(close)
        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False
        hwm = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                valid = (
                    fresh[i] > 0.5
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                    and not np.isnan(ma[i])
                    and close[i] > ma[i]
                )
                if valid:
                    in_pos = True
                    entry_price = close[i]
                    stop = close[i] - params.init_stop_k * atr[i]
                    breakeven_done = False
                    hwm = close[i]
                    bars_held = 0
                    sig[i] = 1
            else:
                bars_held += 1
                if close[i] > hwm:
                    hwm = close[i]

                if (not breakeven_done) and close[i] >= entry_price * (1.0 + params.breakeven_target):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_done = True

                if breakeven_done and not np.isnan(atr[i]):
                    trail = hwm - params.trail_k * atr[i]
                    if trail > stop:
                        stop = trail

                exit_now = close[i] < stop or bars_held >= params.max_hold
                if exit_now:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
