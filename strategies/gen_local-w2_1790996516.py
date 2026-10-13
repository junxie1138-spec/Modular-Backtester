from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CentroidParams:
    atr_window: int = 10
    centroid_window: int = 30
    cooling_window: int = 3
    selloff_window: int = 5
    entry_discount: float = 1.5
    cooling_ratio: float = 0.98
    profit_target: float = 0.025
    max_hold_bars: int = 10
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[CentroidParams]):
    strategy_id = "gen_local-w2_1790996516"

    @classmethod
    def params_type(cls) -> type[CentroidParams]:
        return CentroidParams

    @staticmethod
    def warmup_bars(params: CentroidParams) -> int:
        return int(max(
            params.atr_window + params.centroid_window,
            params.atr_window + params.cooling_window + 1,
            params.selloff_window + 1,
        ) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CentroidParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().where(lambda s: s > 0)
        weight = 1.0 / atr.pow(2)
        denominator = weight.rolling(
            params.centroid_window, min_periods=params.centroid_window
        ).sum()
        numerator = (close * weight).rolling(
            params.centroid_window, min_periods=params.centroid_window
        ).sum()
        centroid = numerator / denominator.where(denominator > 0)
        result = pd.DataFrame(index=data.index)
        result["true_range"] = true_range
        result["atr"] = atr
        result["weight"] = weight
        result["weight_sum"] = denominator
        result["weighted_close_sum"] = numerator
        result["centroid"] = centroid
        result["discount"] = (centroid - close) / atr
        result["cooling"] = atr / atr.shift(params.cooling_window)
        result["selloff"] = close / close.shift(params.selloff_window).where(
            close.shift(params.selloff_window) > 0
        ) - 1.0
        result["strength"] = (
            (result["discount"] - params.entry_discount) / params.entry_discount
        ).clip(0.0, 1.0)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CentroidParams,
    ) -> SignalFrame:
        ready = indicators[["discount", "cooling", "selloff", "strength"]].notna().all(axis=1)
        eligible = (
            ready
            & (indicators["discount"] >= params.entry_discount)
            & (indicators["cooling"] <= params.cooling_ratio)
            & (indicators["selloff"] < 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        strengths = (
            params.min_size
            + (1.0 - params.min_size) * indicators["strength"].fillna(0.0)
        ).clip(params.min_size, 1.0).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        count = len(data)
        raw_signal = np.zeros(count, dtype=int)
        raw_size = np.full(count, params.min_size, dtype=float)
        active = False
        fill_bar = -1
        entry_price = np.nan
        entry_size = params.min_size

        for i in range(count):
            if active:
                # The prior close's decision fills at this session's open.
                if i == fill_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                age = i - fill_bar + 1
                profit_reached = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Close-observed exits also execute on the following bar.
                if profit_reached or age >= params.max_hold_bars:
                    active = False
                    continue
            elif eligible[i] and i + 1 < count:
                active = True
                fill_bar = i + 1
                entry_size = strengths[i]

            if active:
                raw_signal[i] = 1
                raw_size[i] = entry_size

        df = pd.DataFrame(
            {"signal": raw_signal, "size": raw_size}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
