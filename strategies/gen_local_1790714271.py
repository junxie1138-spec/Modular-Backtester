from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 12
    degree_window: int = 20
    position_band: float = 0.75


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714271"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return params.graph_window + params.degree_window

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        w = params.graph_window
        degree = pd.Series(np.nan, index=data.index, dtype=float)
        values = close.to_numpy()
        for end in range(w - 1, len(values)):
            segment = values[end - w + 1:end + 1]
            last = segment[-1]
            visible = 1
            for start in range(w - 1):
                slope = (last - segment[start]) / (w - 1 - start)
                if np.all(segment[start + 1:-1] < segment[start] + slope * np.arange(1, w - 1 - start)):
                    visible += 1
            degree.iloc[end] = visible
        lo = close.rolling(params.degree_window, min_periods=params.degree_window).min()
        hi = close.rolling(params.degree_window, min_periods=params.degree_window).max()
        return pd.DataFrame({
            "degree": degree,
            "degree_base": degree.rolling(params.degree_window, min_periods=params.degree_window).median(),
            "position": (close - lo) / (hi - lo).replace(0, np.nan),
        }, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        ready = indicators[["degree", "degree_base", "position"]].notna().all(axis=1)
        topology = indicators["degree"] > indicators["degree_base"]
        long_entry = ready & topology & (indicators["position"] >= params.position_band)
        short_entry = ready & topology & (indicators["position"] <= 1.0 - params.position_band)
        raw = np.zeros(len(data), dtype=int)
        state = 0
        for i, (go_long, go_short) in enumerate(zip(long_entry.to_numpy(), short_entry.to_numpy())):
            if state == 1 and go_short:
                state = 0
            elif state == -1 and go_long:
                state = 0
            elif state == 0:
                if go_long:
                    state = 1
                elif go_short:
                    state = -1
            raw[i] = state
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
