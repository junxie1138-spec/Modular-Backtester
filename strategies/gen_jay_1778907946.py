from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 7
    polarity_thresh: float = 0.25
    atr_vol_min: float = 0.003


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778907946"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # ATR uses close.shift(1), so needs lookback + 1 bars total
        return params.lookback + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        # Intrabar polarity: direction + position of close within bar range, [-1, 1]
        bar_range = (high - low).where((high - low) > 0)
        polarity = (close - open_) / bar_range

        mean_polarity = polarity.rolling(params.lookback).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.lookback).mean()
        atr_ratio = atr / close

        ind = pd.DataFrame(index=data.index)
        ind["mean_polarity"] = mean_polarity
        ind["atr_ratio"] = atr_ratio
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        mp_arr = indicators["mean_polarity"].to_numpy(dtype=float)
        ar_arr = indicators["atr_ratio"].to_numpy(dtype=float)

        n = len(mp_arr)
        sig = np.zeros(n, dtype=int)
        current = 0
        thresh = params.polarity_thresh
        vol_min = params.atr_vol_min

        for i in range(n):
            m = mp_arr[i]
            ar = ar_arr[i]

            if np.isnan(m) or np.isnan(ar):
                current = 0
                sig[i] = 0
                continue

            # Only update state when volatility is sufficient
            if ar >= vol_min:
                if m > thresh:
                    current = 1
                elif m < -thresh:
                    current = -1
                # deadband [-thresh, thresh]: hold current position (hysteresis)

            sig[i] = current

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
