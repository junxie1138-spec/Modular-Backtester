from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRotationParams:
    atr_window: int = 14
    area_window: int = 4
    short_window: int = 3
    long_window: int = 20
    area_threshold: float = 0.03
    change_threshold: float = 0.6
    stop_atr: float = 1.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GapRotationParams]):
    strategy_id = "gen_local-w2_1791297547"

    @classmethod
    def params_type(cls) -> type[GapRotationParams]:
        return GapRotationParams

    @staticmethod
    def warmup_bars(params: GapRotationParams) -> int:
        return params.atr_window + max(params.long_window, params.area_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRotationParams) -> pd.DataFrame:
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
        denominator = atr.replace(0, np.nan)
        gap = ((data["open"] - previous_close) / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        session = ((data["close"] - data["open"]) / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        area = pd.Series(0.0, index=data.index)
        for older in range(params.area_window - 1, 0, -1):
            for newer in range(older - 1, -1, -1):
                area = area + 0.5 * (
                    gap.shift(older) * session.shift(newer)
                    - session.shift(older) * gap.shift(newer)
                )
        pairs = params.area_window * (params.area_window - 1) / 2
        area = (area / pairs).fillna(0.0)

        spread = session - gap
        short_mean = spread.rolling(params.short_window, min_periods=params.short_window).mean()
        long_mean = spread.rolling(params.long_window, min_periods=params.long_window).mean()
        long_std = spread.rolling(params.long_window, min_periods=params.long_window).std().replace(0, np.nan)
        change_z = ((short_mean - long_mean) / long_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({"atr": atr.fillna(0.0), "area": area, "change_z": change_z}, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRotationParams,
    ) -> SignalFrame:
        z = indicators["change_z"]
        entries = (
            (z > params.change_threshold)
            & (z.shift(1) <= params.change_threshold)
            & (indicators["area"] < -params.area_threshold)
        ).fillna(False).to_numpy()
        closes = data["close"].to_numpy()
        atrs = indicators["atr"].to_numpy()
        target = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = 0
        peak = -np.inf
        stop = -np.inf
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            if holding:
                if i >= entry_bar:
                    peak = max(peak, closes[i])
                    stop = max(stop, peak - params.stop_atr * atrs[i])
                    if closes[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                        holding = False
                target[i] = int(holding)
            elif i >= warmup and entries[i]:
                holding = True
                entry_bar = i + 1
                peak = -np.inf
                stop = -np.inf
                target[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
