from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 20
    z_lookback: int = 20
    price_z_thresh: float = 1.5
    vol_z_period: int = 10
    vol_z_thresh: float = 1.0
    atr_period: int = 14
    trail_k: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778887613"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.ma_period, params.z_lookback, params.vol_z_period, params.atr_period) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        sma = close.rolling(params.ma_period).mean()
        roll_std = close.rolling(params.z_lookback).std()
        ind["price_z"] = (close - sma) / roll_std.where(roll_std > 0)

        vol_sma = volume.rolling(params.vol_z_period).mean()
        vol_std = volume.rolling(params.vol_z_period).std()
        ind["vol_z"] = (volume - vol_sma) / vol_std.where(vol_std > 0)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        price_z = indicators["price_z"].values
        vol_z = indicators["vol_z"].values
        atr = indicators["atr"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_trade = False
        hwm = 0.0

        for i in range(n):
            if np.isnan(price_z[i]) or np.isnan(vol_z[i]) or np.isnan(atr[i]):
                signal[i] = 0
                continue

            if in_trade:
                if close[i] > hwm:
                    hwm = close[i]
                if close[i] < hwm - params.trail_k * atr[i]:
                    signal[i] = 0
                    in_trade = False
                else:
                    signal[i] = 1
            else:
                if price_z[i] > params.price_z_thresh and vol_z[i] > params.vol_z_thresh:
                    signal[i] = 1
                    in_trade = True
                    hwm = close[i]

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
