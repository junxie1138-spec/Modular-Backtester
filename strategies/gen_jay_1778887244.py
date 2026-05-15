from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 14
    z_threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778887244"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 2 * params.window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        w = params.window
        w2 = 2 * w

        ma_s = close.rolling(w).mean()
        std_s = close.rolling(w).std(ddof=0)
        z_short = (close - ma_s) / std_s.where(std_s > 0, np.nan)

        ma_l = close.rolling(w2).mean()
        std_l = close.rolling(w2).std(ddof=0)
        z_long = (close - ma_l) / std_l.where(std_l > 0, np.nan)

        range_s = data["high"].rolling(w).max() - data["low"].rolling(w).min()
        range_l = data["high"].rolling(w2).max() - data["low"].rolling(w2).min()
        range_ratio = range_s / range_l.where(range_l > 0, np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["z_short"] = z_short
        ind["z_long"] = z_long
        ind["range_ratio"] = range_ratio
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        z_short = indicators["z_short"].values
        z_long = indicators["z_long"].values
        range_ratio = indicators["range_ratio"].values
        threshold = params.z_threshold

        n = len(z_short)
        raw = np.zeros(n, dtype=np.int64)
        in_trade = False

        for i in range(n):
            zs = z_short[i]
            zl = z_long[i]
            rr = range_ratio[i]

            if np.isnan(zs) or np.isnan(zl) or np.isnan(rr):
                continue

            if in_trade:
                if zs > 0.0:
                    in_trade = False
                else:
                    raw[i] = 1
            else:
                if zs < -threshold and zl < 0.0 and rr < 0.5:
                    in_trade = True
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
