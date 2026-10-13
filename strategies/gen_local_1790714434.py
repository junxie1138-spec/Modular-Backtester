from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    volume_mult: float = 1.5
    breakout_window: int = 5
    atr_mult: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714434"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(200, 20, params.breakout_window, 14)

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
        return pd.DataFrame(
            {
                "ma200": data["close"].rolling(200, min_periods=200).mean(),
                "atr": tr.rolling(14, min_periods=14).mean(),
                "volume_mean": data["volume"].rolling(20, min_periods=20).mean(),
                "prior_high": data["high"].shift(1).rolling(
                    params.breakout_window, min_periods=params.breakout_window
                ).max(),
                "prior_low": data["low"].shift(1).rolling(
                    params.breakout_window, min_periods=params.breakout_window
                ).min(),
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        volume_ok = data["volume"] >= indicators["volume_mean"] * params.volume_mult
        valid = indicators[["ma200", "atr", "prior_high", "prior_low", "volume_mean"]].notna().all(axis=1)
        long_entry = valid & volume_ok & (data["close"] > indicators["prior_high"]) & (data["close"] > indicators["ma200"])
        short_entry = valid & volume_ok & (data["close"] < indicators["prior_low"]) & (data["close"] < indicators["ma200"])

        signal = np.zeros(len(data), dtype=int)
        position = 0
        extreme = 0.0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = indicators["atr"].iloc[i]
            if position == 0:
                if long_entry.iloc[i]:
                    position = 1
                    extreme = close
                elif short_entry.iloc[i]:
                    position = -1
                    extreme = close
            elif pd.notna(atr):
                if position == 1:
                    extreme = max(extreme, close)
                    if close <= extreme - params.atr_mult * float(atr):
                        position = 0
                else:
                    extreme = min(extreme, close)
                    if close >= extreme + params.atr_mult * float(atr):
                        position = 0
            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
