from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_window: int = 10
    vol_rank_window: int = 60
    max_vol_percentile: float = 0.30
    arm_day: int = 24
    trigger_day: int = 25
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712925"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.vol_window + params.vol_rank_window

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        log_return = np.log(data["close"] / data["close"].shift(1))
        out["volatility"] = log_return.rolling(params.vol_window, min_periods=params.vol_window).std()
        out["vol_threshold"] = out["volatility"].rolling(
            params.vol_rank_window, min_periods=params.vol_rank_window
        ).quantile(params.max_vol_percentile)
        out["atr"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - data["close"].shift(1)).abs(),
                (data["low"] - data["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1).rolling(params.vol_window, min_periods=params.vol_window).mean()
        out["entry"] = (
            (out["volatility"] <= out["vol_threshold"])
            & (data.index.day >= params.trigger_day)
            & (data.index.day >= params.arm_day)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = indicators["entry"].to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        in_position = False
        entry_price = 0.0
        stop = 0.0
        for i in range(len(data)):
            if in_position:
                if np.isfinite(atr[i]):
                    if high[i] >= entry_price * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry_price, high[i] - params.trail_atr * atr[i])
                    elif stop < entry_price:
                        stop = entry_price - params.trail_atr * atr[i]
                if low[i] <= stop:
                    in_position = False
                    continue
                signal[i] = 1
            elif raw[i] and np.isfinite(atr[i]) and close[i] > 0:
                in_position = True
                entry_price = close[i]
                stop = entry_price - params.trail_atr * atr[i]
                signal[i] = 1
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
