from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    range_period: int = 20
    vol_lookback: int = 60
    vol_compress_thresh: float = 0.25
    pos_entry_thresh: float = 0.65
    trail_k: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778897402"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.atr_period + params.vol_lookback, params.range_period) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_period).mean()
        safe_close = close.where(close > 0, other=np.nan)
        atr_pct = atr / safe_close

        vol_rank = atr_pct.rolling(params.vol_lookback).rank(pct=True)

        roll_high = close.rolling(params.range_period).max()
        roll_low = close.rolling(params.range_period).min()
        range_width = roll_high - roll_low
        safe_width = range_width.where(range_width > 0, other=np.nan)
        pos_in_range = (close - roll_low) / safe_width

        return pd.DataFrame(
            {
                "atr": atr,
                "vol_rank": vol_rank,
                "pos_in_range": pos_in_range,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = data["close"].to_numpy()
        atr_arr = indicators["atr"].to_numpy()
        vol_rank_arr = indicators["vol_rank"].to_numpy()
        pos_in_range_arr = indicators["pos_in_range"].to_numpy()

        n = len(close_arr)
        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_trade = False
        hwm = 0.0
        entry_size = 1.0

        for i in range(n):
            vr = vol_rank_arr[i]
            pir = pos_in_range_arr[i]
            a = atr_arr[i]
            c = close_arr[i]

            if np.isnan(vr) or np.isnan(pir) or np.isnan(a):
                if in_trade:
                    signal[i] = 1
                    size[i] = entry_size
                continue

            if not in_trade:
                if vr < params.vol_compress_thresh and pir > params.pos_entry_thresh:
                    compression_score = 1.0 - vr
                    position_score = pir
                    entry_size = float(
                        np.clip(compression_score * position_score * 2.0, 0.1, 1.0)
                    )
                    signal[i] = 1
                    size[i] = entry_size
                    in_trade = True
                    hwm = c
            else:
                if c > hwm:
                    hwm = c
                stop_level = hwm - params.trail_k * a
                if c < stop_level:
                    signal[i] = 0
                    in_trade = False
                    hwm = 0.0
                else:
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
