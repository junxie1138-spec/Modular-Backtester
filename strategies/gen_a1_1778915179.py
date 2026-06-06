from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    ma_window: int = 200
    rank_window: int = 20
    force_window: int = 60
    vel_lag: int = 3
    force_thresh: float = 0.80
    breakeven_pct: float = 0.015
    atr_window: int = 14
    k_init: float = 2.5
    k_trail: float = 1.5


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778915179"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    @staticmethod
    def warmup_bars(params: GenA1Params) -> int:
        spans = [
            params.ma_window,
            params.rank_window + params.force_window,
            params.rank_window + params.vel_lag,
            params.atr_window + 1,
        ]
        return int(max(spans) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()

        rank = close.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        force = rank * (1.0 - rank)
        force_pct = force.rolling(params.force_window, min_periods=params.force_window).rank(pct=True)
        rank_vel = rank - rank.shift(params.vel_lag)

        ind = pd.DataFrame(index=data.index)
        ind["ma"] = ma
        ind["atr"] = atr
        ind["rank"] = rank
        ind["force"] = force
        ind["force_pct"] = force_pct
        ind["rank_vel"] = rank_vel
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        ma = indicators["ma"].to_numpy(dtype=float)
        atr = np.nan_to_num(indicators["atr"].to_numpy(dtype=float), nan=0.0)
        force_pct = np.nan_to_num(indicators["force_pct"].to_numpy(dtype=float), nan=0.0)
        rank_vel = np.nan_to_num(indicators["rank_vel"].to_numpy(dtype=float), nan=-1.0)

        n = len(close)

        regime = close > ma
        entry_raw = (
            regime
            & (force_pct >= params.force_thresh)
            & (rank_vel > 0.0)
            & (atr > 0.0)
        )

        pos = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        stop = 0.0
        be_done = False

        for i in range(n):
            if not in_pos:
                if entry_raw[i]:
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - params.k_init * atr[i]
                    be_done = False
                    pos[i] = 1
            else:
                if (not be_done) and high[i] >= entry_price * (1.0 + params.breakeven_pct):
                    if entry_price > stop:
                        stop = entry_price
                    be_done = True
                if be_done and atr[i] > 0.0:
                    trail = close[i] - params.k_trail * atr[i]
                    if trail > stop:
                        stop = trail
                if close[i] <= stop or low[i] <= stop:
                    pos[i] = 0
                    in_pos = False
                else:
                    pos[i] = 1

        signal = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)
        size = pd.Series(0.6 + 0.8 * force_pct, index=data.index).clip(lower=0.1).astype(float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
