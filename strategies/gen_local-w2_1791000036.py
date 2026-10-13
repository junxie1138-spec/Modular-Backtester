from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureCompressionParams:
    signature_window: int = 6
    compression_window: int = 5
    baseline_window: int = 40
    compression_ratio: float = 0.75
    area_threshold: float = 0.08
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SignatureCompressionParams]):
    strategy_id = "gen_local-w2_1791000036"

    @classmethod
    def params_type(cls) -> type[SignatureCompressionParams]:
        return SignatureCompressionParams

    def warmup_bars(self, params: SignatureCompressionParams) -> int:
        return int(max(max(2, params.signature_window) + 1,
                       max(2, params.compression_window),
                       max(2, params.baseline_window)) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureCompressionParams) -> pd.DataFrame:
        window = max(2, int(params.signature_window))
        short_window = max(2, int(params.compression_window))
        baseline_window = max(2, int(params.baseline_window))
        close = data["close"].astype(float).where(data["close"] > 0)
        relative_range = ((data["high"] - data["low"]) / close).where(data["high"] > data["low"])
        x = np.log(close)
        y = np.log(relative_range)
        dx = x.diff()
        dy = y.diff()
        cross = x.shift(1) * dy - y.shift(1) * dx
        sum_dx = dx.rolling(window, min_periods=window).sum()
        sum_dy = dy.rolling(window, min_periods=window).sum()
        area = 0.5 * (cross.rolling(window, min_periods=window).sum()
                      - x.shift(window) * sum_dy
                      + y.shift(window) * sum_dx)
        path_scale = (dx.abs().rolling(window, min_periods=window).sum()
                      * dy.abs().rolling(window, min_periods=window).sum())
        area_score = area / path_scale.where(path_scale > 0)
        short_range = relative_range.rolling(short_window, min_periods=short_window).mean()
        baseline_range = relative_range.rolling(baseline_window, min_periods=baseline_window).mean()
        compression = short_range / baseline_range.where(baseline_range > 0)
        out = pd.DataFrame({
            "log_price": x,
            "log_range": y,
            "price_increment": dx,
            "range_increment": dy,
            "segment_cross": cross,
            "price_displacement": sum_dx,
            "range_displacement": sum_dy,
            "signature_area": area,
            "path_scale": path_scale,
            "area_score": area_score,
            "short_range": short_range,
            "baseline_range": baseline_range,
            "compression": compression,
        }, index=data.index)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out[["area_score", "compression", "price_displacement"]].notna().all(axis=1)
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureCompressionParams) -> SignalFrame:
        compressed = indicators["ready"].astype(bool) & (
            indicators["compression"] <= params.compression_ratio)
        threshold = max(0.0, float(params.area_threshold))
        long_setup = compressed & (indicators["area_score"] > threshold) & (
            indicators["price_displacement"] > 0)
        short_setup = compressed & (indicators["area_score"] < -threshold) & (
            indicators["price_displacement"] < 0)
        confirmed_long = long_setup & long_setup.shift(1, fill_value=False)
        confirmed_short = short_setup & short_setup.shift(1, fill_value=False)
        requests = np.where(confirmed_long, 1, np.where(confirmed_short, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        holding_bars = max(1, int(params.holding_bars))
        position = 0
        entry_bar = -1
        for i, request in enumerate(requests):
            if position != 0:
                if i - entry_bar >= holding_bars:
                    position = 0
            elif request != 0:
                position = int(request)
                entry_bar = i
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
