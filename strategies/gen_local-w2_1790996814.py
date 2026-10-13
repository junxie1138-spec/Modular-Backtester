from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticInnovationParams:
    slope_window: int = 8
    correlation_window: int = 40
    percentile_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    trailing_atr: float = 1.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[ElasticInnovationParams]):
    strategy_id = "gen_local-w2_1790996814"

    @classmethod
    def params_type(cls) -> type[ElasticInnovationParams]:
        return ElasticInnovationParams

    @staticmethod
    def warmup_bars(params: ElasticInnovationParams) -> int:
        return int(max(
            max(2, params.slope_window)
            + max(3, params.correlation_window)
            + max(2, params.percentile_window) + 3,
            max(2, params.atr_window) + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ElasticInnovationParams) -> pd.DataFrame:
        slope_window = max(2, int(params.slope_window))
        correlation_window = max(3, int(params.correlation_window))
        percentile_window = max(2, int(params.percentile_window))
        atr_window = max(2, int(params.atr_window))
        percentile = float(np.clip(params.entry_percentile, 0.01, 0.99))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_return"] = np.log(close.where(close > 0)).diff()
        out["latent_slope"] = out["log_return"].rolling(
            slope_window, min_periods=slope_window
        ).mean()
        out["innovation"] = out["latent_slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["innovation_correlation"] = out["innovation"].rolling(
            correlation_window, min_periods=correlation_window
        ).corr(out["lagged_innovation"]).clip(-1.0, 1.0)
        out["innovation_scale"] = out["innovation"].rolling(
            correlation_window, min_periods=correlation_window
        ).std(ddof=0).replace(0.0, np.nan)
        out["rebound_score"] = (
            out["innovation_correlation"] * out["innovation"]
            / out["innovation_scale"]
        )
        out["entry_threshold"] = out["rebound_score"].shift(1).rolling(
            percentile_window, min_periods=percentile_window
        ).quantile(percentile)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            atr_window, min_periods=atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticInnovationParams,
    ) -> SignalFrame:
        required = [
            "latent_slope", "innovation", "innovation_correlation",
            "rebound_score", "entry_threshold", "atr",
        ]
        ready = indicators[required].notna().all(axis=1)
        entry = (
            ready
            & (indicators["latent_slope"] > 0.0)
            & (indicators["innovation"] < 0.0)
            & (indicators["innovation_correlation"] < 0.0)
            & (indicators["rebound_score"] > indicators["entry_threshold"])
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        holding_bars = min(2, max(1, int(params.holding_bars)))
        trailing_atr = max(0.01, float(params.trailing_atr))
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            if active:
                # The preceding decision enters at this bar's open.
                age += 1
                if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0):
                    active = False
                    continue
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - trailing_atr * atr[i])
                if close[i] <= stop or age >= holding_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entry[i] and np.isfinite(close[i]) and close[i] > 0.0:
                active = True
                age = 0
                # Initialize using the first close after the actual entry.
                high_water = -np.inf
                stop = -np.inf
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
