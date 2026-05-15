from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    cum_ret_window: int = 20
    rank_window: int = 60
    snr_window: int = 10
    rank_threshold: float = 0.70
    snr_threshold: float = 0.30
    atr_window: int = 14
    atr_mult: float = 2.0
    snr_scale: float = 1.0
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778887998"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + params.cum_ret_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        daily_ret = close.pct_change()
        cum_ret = close / close.shift(params.cum_ret_window) - 1
        rank = cum_ret.rolling(params.rank_window).rank(pct=True)

        roll_mean = daily_ret.rolling(params.snr_window).mean()
        roll_std = daily_ret.rolling(params.snr_window).std()
        snr = roll_mean / (roll_std + 1e-9)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ind = pd.DataFrame(index=data.index)
        ind["rank"] = rank
        ind["snr"] = snr
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
        rank = indicators["rank"].values
        snr = indicators["snr"].values
        atr = indicators["atr"].values
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.base_size)

        in_trade = False
        hwm = 0.0
        trade_snr = 0.0

        for i in range(n):
            r = rank[i]
            s = snr[i]
            a = atr[i]
            c = close[i]

            if np.isnan(r) or np.isnan(s) or np.isnan(a):
                continue

            if in_trade:
                if c > hwm:
                    hwm = c
                if c < hwm - params.atr_mult * a:
                    in_trade = False
                else:
                    raw_signal[i] = 1
                    raw_size[i] = np.clip(abs(trade_snr) / params.snr_scale, 0.5, 1.0) * params.base_size
            else:
                if r > params.rank_threshold and s > params.snr_threshold:
                    in_trade = True
                    hwm = c
                    trade_snr = s
                    raw_signal[i] = 1
                    raw_size[i] = np.clip(abs(s) / params.snr_scale, 0.5, 1.0) * params.base_size

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
