from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_lookback: int = 20
    min_squeeze_bars: int = 3
    min_dir_streak: int = 3
    atr_lookback: int = 14
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778884174"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.range_lookback, params.atr_lookback) + params.min_squeeze_bars + params.min_dir_streak + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_lookback).mean()

        daily_range = data["high"] - data["low"]
        range_median = daily_range.rolling(params.range_lookback).median()
        is_compressed = (daily_range < range_median).fillna(False)

        grp_c = (is_compressed != is_compressed.shift()).cumsum()
        cs = is_compressed.groupby(grp_c).cumcount() + 1
        ind["compression_streak"] = cs.where(is_compressed, 0).astype(float)

        bearish = (data["close"] < data["open"]).fillna(False)
        grp_b = (bearish != bearish.shift()).cumsum()
        sb = bearish.groupby(grp_b).cumcount() + 1
        ind["bear_streak"] = sb.where(bearish, 0).astype(float)

        bullish = (data["close"] > data["open"]).fillna(False)
        grp_u = (bullish != bullish.shift()).cumsum()
        su = bullish.groupby(grp_u).cumcount() + 1
        ind["bull_streak"] = su.where(bullish, 0).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        compression = indicators["compression_streak"].values
        bear_streak = indicators["bear_streak"].values
        bull_streak = indicators["bull_streak"].values

        n = len(close)
        signal_arr = np.zeros(n, dtype=np.int64)

        position = 0
        hwm = np.nan
        lwm = np.nan

        for i in range(n):
            if np.isnan(atr[i]):
                continue

            if position == 0:
                sq = compression[i]
                bs = bear_streak[i]
                us = bull_streak[i]
                if sq >= params.min_squeeze_bars and bs >= params.min_dir_streak:
                    position = 1
                    hwm = close[i]
                    signal_arr[i] = 1
                elif sq >= params.min_squeeze_bars and us >= params.min_dir_streak:
                    position = -1
                    lwm = close[i]
                    signal_arr[i] = -1

            elif position == 1:
                if close[i] > hwm:
                    hwm = close[i]
                if close[i] < hwm - params.atr_mult * atr[i]:
                    position = 0
                    hwm = np.nan
                    signal_arr[i] = 0
                else:
                    signal_arr[i] = 1

            else:
                if close[i] < lwm:
                    lwm = close[i]
                if close[i] > lwm + params.atr_mult * atr[i]:
                    position = 0
                    lwm = np.nan
                    signal_arr[i] = 0
                else:
                    signal_arr[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal_arr, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
