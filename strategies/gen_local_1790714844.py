from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 10
    pressure_span: int = 5
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714844"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.graph_window + 60, 15, 4)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        # Natural visibility degree of the newest point in each trailing window.
        values = close.to_numpy(dtype=float)
        degree = np.full(len(values), np.nan, dtype=float)
        width = params.graph_window
        for i in range(width - 1, len(values)):
            start = i - width + 1
            endpoint = values[i]
            count = 0
            for j in range(start, i):
                visible = True
                slope = (endpoint - values[j]) / (i - j)
                for k in range(j + 1, i):
                    if values[k] >= values[j] + slope * (k - j):
                        visible = False
                        break
                count += int(visible)
            degree[i] = count

        degree_series = pd.Series(degree, index=data.index)
        degree_floor = degree_series.rolling(60, min_periods=60).quantile(0.25)
        downside = (-close.pct_change()).clip(lower=0.0)
        pressure = downside.ewm(span=params.pressure_span, adjust=False, min_periods=params.pressure_span).mean()
        return pd.DataFrame(
            {
                "atr": atr,
                "degree": degree_series,
                "degree_floor": degree_floor,
                "pressure": pressure,
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
        degree = indicators["degree"]
        pressure = indicators["pressure"]
        setup = (
            degree.gt(indicators["degree"].shift(1))
            & degree.shift(1).le(indicators["degree_floor"].shift(1))
            & pressure.lt(pressure.shift(1))
            & pressure.shift(1).lt(pressure.shift(2))
        ).fillna(False).to_numpy()

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        held = False
        entry = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            if held:
                age += 1
                if close[i] < stop or age >= 2:
                    held = False
                else:
                    signal[i] = 1
            if not held and setup[i] and np.isfinite(atr[i]):
                held = True
                entry = close[i]
                stop = entry - params.stop_atr * atr[i]
                age = 0
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
