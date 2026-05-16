from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_period: int = 5
    atr_period: int = 14
    season_window: int = 8
    accel_threshold: float = 0.0
    breakeven_pct: float = 0.015
    trail_k: float = 1.5
    min_accel_rank: float = 0.55


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778904987"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.season_window * 26 + max(params.atr_period, params.roc_period + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        dow = pd.Series(data.index.dayofweek, index=data.index)
        wom = pd.Series(np.clip((data.index.day - 1) // 7, 0, 4), index=data.index)
        cell = (dow * 5 + wom).astype(int)

        seasonal_accel = pd.Series(np.nan, index=data.index)
        for cell_id in range(25):
            mask = cell == cell_id
            if mask.sum() < 3:
                continue
            cell_vals = roc_accel[mask]
            rolling_mean = cell_vals.rolling(params.season_window, min_periods=3).mean()
            seasonal_accel[mask] = rolling_mean.values

        accel_rank = roc_accel.rolling(252, min_periods=30).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["roc_accel"] = roc_accel
        ind["seasonal_accel"] = seasonal_accel
        ind["accel_rank"] = accel_rank
        ind["atr"] = atr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr_vals = indicators["atr"].values
        seasonal_accel = indicators["seasonal_accel"].values
        accel_rank = indicators["accel_rank"].values
        n = len(close)

        entry_mask = (
            np.isfinite(seasonal_accel)
            & np.isfinite(accel_rank)
            & np.isfinite(atr_vals)
            & (seasonal_accel > params.accel_threshold)
            & (accel_rank >= params.min_accel_rank)
        )

        lo = max(params.min_accel_rank, 1e-8)
        hi_range = max(1.0 - lo, 1e-8)
        raw_size = np.where(
            np.isfinite(accel_rank),
            0.5 + 0.5 * np.clip((accel_rank - lo) / hi_range, 0.0, 1.0),
            0.5,
        )

        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.5)

        in_position = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_size = 0.5

        for i in range(n):
            can_enter = True
            if in_position:
                if close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    trail_stop = max(trail_stop, entry_price)
                if np.isfinite(atr_vals[i]) and atr_vals[i] > 0.0:
                    trail_stop = max(trail_stop, close[i] - params.trail_k * atr_vals[i])
                if close[i] <= trail_stop:
                    in_position = False
                    signal[i] = 0
                    can_enter = False
                else:
                    signal[i] = 1
                    size[i] = entry_size

            if not in_position and can_enter and entry_mask[i]:
                in_position = True
                entry_price = close[i]
                entry_size = float(raw_size[i])
                trail_stop = close[i] - params.trail_k * atr_vals[i]
                signal[i] = 1
                size[i] = entry_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.5)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
