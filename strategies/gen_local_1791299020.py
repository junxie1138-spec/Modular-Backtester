from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphTriangleParams:
    span: int = 6
    trigger: int = 2
    min_size: float = 0.30


class GeneratedStrategy(BaseStrategy[GraphTriangleParams]):
    strategy_id = "gen_local_1791299020"

    @classmethod
    def params_type(cls) -> type[GraphTriangleParams]:
        return GraphTriangleParams

    @staticmethod
    def warmup_bars(params: GraphTriangleParams) -> int:
        return params.span

    @staticmethod
    def indicators(data: pd.DataFrame, params: GraphTriangleParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        counts = {}
        for direction, name in ((1, "up_triangles"), (-1, "down_triangles")):
            price = direction * close
            visible = {1: price.notna() & price.shift(1).notna()}
            count = pd.Series(0, index=data.index, dtype=int)
            for lag in range(2, params.span):
                edge = price.notna() & price.shift(lag).notna()
                for middle in range(1, lag):
                    chord = price + (price.shift(lag) - price) * middle / lag
                    edge &= price.shift(middle) < chord
                visible[lag] = edge
                triangle = (
                    edge
                    & visible[lag - 1].shift(1).eq(True)
                    & price.gt(price.shift(1))
                    & price.gt(price.shift(lag))
                )
                count += triangle.astype(int)
            counts[name] = count
        result = pd.DataFrame(counts, index=data.index)
        result["score"] = result["up_triangles"] - result["down_triangles"]
        result["ready"] = np.arange(len(data)) >= params.span - 1
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GraphTriangleParams,
    ) -> SignalFrame:
        bullish = (indicators["score"] >= params.trigger) & indicators["ready"]
        bearish = (indicators["score"] <= -params.trigger) & indicators["ready"]
        position = 0
        positions = []
        for enter, exit_ in zip(bullish, bearish):
            if position == 0 and enter:
                position = 1
            elif position == 1 and exit_:
                position = 0
            positions.append(position)
        strength = (indicators["up_triangles"] / (params.span - 2)).clip(0, 1)
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (params.min_size + (0.95 - params.min_size) * strength).shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
