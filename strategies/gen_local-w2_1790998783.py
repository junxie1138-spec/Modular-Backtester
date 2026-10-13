from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnergyParams:
    energy_window: int = 12
    percentile_window: int = 80
    entry_percentile: float = 0.75
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapEnergyParams]):
    strategy_id = "gen_local-w2_1790998783"

    @classmethod
    def params_type(cls) -> type[GapEnergyParams]:
        return GapEnergyParams

    def warmup_bars(self, params: GapEnergyParams) -> int:
        return int(max(8 + params.energy_window + params.percentile_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapEnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).replace(0.0, np.nan)
        gap = data["open"].div(previous) - 1.0
        out["gap"] = gap
        for scale in (1, 2, 4):
            block = gap.rolling(scale, min_periods=scale).sum()
            detail = (block - block.shift(scale)) / np.sqrt(2.0 * scale)
            out[f"detail_{scale}"] = detail
            out[f"energy_{scale}"] = detail.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["fine_share"] = out["energy_1"].div(total.replace(0.0, np.nan))
        out["fine_threshold"] = out["fine_share"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile).shift(1)
        out["gap_threshold"] = gap.abs().rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile).shift(1)
        out["recovery"] = data["close"].div(previous) - 1.0
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEnergyParams,
    ) -> SignalFrame:
        ready = indicators[[
            "gap", "gap_threshold", "fine_share", "fine_threshold", "recovery", "atr"
        ]].notna().all(axis=1) & indicators["atr"].gt(0.0)
        entries = (
            ready
            & indicators["gap"].lt(-indicators["gap_threshold"])
            & indicators["fine_share"].gt(indicators["fine_threshold"])
            & indicators["recovery"].ge(0.0)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    active = False
                else:
                    if age == 1:
                        # First held bar: execution occurred at this bar's open.
                        high_water = closes[i]
                        stop = opens[i] - params.trail_atr * atr[i - 1]
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if closes[i] <= stop or age >= params.max_hold:
                        active = False
                # Exit decisions cannot re-enter on the same close.
            elif entries[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = np.nan
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
