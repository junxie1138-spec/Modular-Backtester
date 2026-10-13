from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MedianReturnParams:
    median_window: int = 18
    history_window: int = 252
    percentile: float = 0.80
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[MedianReturnParams]):
    strategy_id = "gen_local_1791226629"

    @classmethod
    def params_type(cls) -> type[MedianReturnParams]:
        return MedianReturnParams

    @classmethod
    def warmup_bars(cls, params: MedianReturnParams) -> int:
        return max(params.median_window + params.history_window + 1, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: MedianReturnParams) -> pd.DataFrame:
        returns = data["close"].pct_change()
        median = returns.rolling(params.median_window, min_periods=params.median_window).median()
        history = median.shift(1).rolling(params.history_window, min_periods=params.history_window)
        upper = history.quantile(params.percentile)
        lower = history.quantile(1.0 - params.percentile)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        valid = median.notna() & upper.notna() & lower.notna() & atr.notna() & (atr > 0)
        return pd.DataFrame(
            {
                "median": median.fillna(0.0),
                "upper": upper.fillna(0.0),
                "lower": lower.fillna(0.0),
                "atr": atr.fillna(0.0),
                "valid": valid.astype(int),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MedianReturnParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        median = indicators["median"].to_numpy(dtype=float)
        upper = indicators["upper"].to_numpy(dtype=float)
        lower = indicators["lower"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = indicators["valid"].to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        extreme = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            if not valid[i] or not np.isfinite(close[i]):
                position = 0
                continue

            if position == 0:
                if median[i] > 0 and median[i] > upper[i]:
                    position = 1
                elif median[i] < 0 and median[i] < lower[i]:
                    position = -1
                held = 0
            else:
                held += 1
                if held == 1:
                    extreme = close[i]
                    stop = extreme - position * params.trail_atr * atr[i]
                elif position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                if held >= params.max_hold or (position == 1 and close[i] <= stop) or (position == -1 and close[i] >= stop):
                    position = 0

            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
