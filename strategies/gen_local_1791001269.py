from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReleaseParams:
    compression_bars: int = 8
    baseline_bars: int = 40
    compression_limit: float = 2.5
    volume_multiple: float = 1.4
    move_atr: float = 0.6
    acceptance_fraction: float = 0.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ReleaseParams]):
    strategy_id = "gen_local_1791001269"

    @classmethod
    def params_type(cls) -> type[ReleaseParams]:
        return ReleaseParams

    def warmup_bars(self, params: ReleaseParams) -> int:
        return int(max(params.compression_bars, params.baseline_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: ReleaseParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        baseline_range = true_range.rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(1).replace(0.0, np.nan)
        baseline_volume = data["volume"].rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(1).replace(0.0, np.nan)
        ceiling = data["high"].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).max().shift(1)
        floor = data["low"].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).min().shift(1)
        body = close - data["open"]
        body_length = body.abs().replace(0.0, np.nan)
        upper_boundary = pd.concat([data["open"], ceiling], axis=1).max(axis=1)
        lower_boundary = pd.concat([data["open"], floor], axis=1).min(axis=1)
        ready = (
            baseline_range.notna() & baseline_volume.notna()
            & ceiling.notna() & floor.notna() & previous_close.notna()
        )
        result = pd.DataFrame({
            "ready": ready,
            "true_range": true_range,
            "baseline_range": baseline_range,
            "baseline_volume": baseline_volume,
            "ceiling": ceiling,
            "floor": floor,
            "compression": (ceiling - floor) / baseline_range,
            "volume_ratio": data["volume"] / baseline_volume,
            "displacement": (close - previous_close) / baseline_range,
            "body": body,
            "upper_acceptance": ((close - upper_boundary).clip(lower=0.0) / body_length).clip(upper=1.0),
            "lower_acceptance": ((lower_boundary - close).clip(lower=0.0) / body_length).clip(upper=1.0),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReleaseParams,
    ) -> SignalFrame:
        x = indicators
        compressed = x["ready"].astype(bool) & (x["compression"] <= params.compression_limit)
        participation = x["volume_ratio"] >= params.volume_multiple
        volume_up = participation & (x["displacement"] >= params.move_atr)
        volume_down = participation & (x["displacement"] <= -params.move_atr)
        geometry_up = (
            compressed & (x["body"] > 0.0)
            & (data["close"] > x["ceiling"])
            & (x["upper_acceptance"] >= params.acceptance_fraction)
        )
        geometry_down = (
            compressed & (x["body"] < 0.0)
            & (data["close"] < x["floor"])
            & (x["lower_acceptance"] >= params.acceptance_fraction)
        )
        candidates = np.select(
            [volume_up & geometry_up, volume_down & geometry_down],
            [1, -1], default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        holding_period = max(1, int(params.hold_bars))
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar == holding_period:
                    position = 0
                    signals[bar] = 0
                    continue
            elif candidates[bar] != 0:
                position = int(candidates[bar])
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
