from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolumeParams:
    history_bars: int = 252
    rank_bars: int = 126
    forecast_bars: int = 7
    pressure_bars: int = 3
    volume_bars: int = 21
    threshold_quantile: float = 0.55


class GeneratedStrategy(BaseStrategy[SeasonalVolumeParams]):
    strategy_id = "gen_local_1791000204"

    @classmethod
    def params_type(cls):
        return SeasonalVolumeParams

    def warmup_bars(self, params: SeasonalVolumeParams) -> int:
        return int(max(
            params.forecast_bars + params.history_bars + params.rank_bars + 1,
            params.volume_bars + params.pressure_bars + params.rank_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: SeasonalVolumeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["phase"] = ((data.index.day - 1) // 7).astype(int)
        out["mature_return"] = close.pct_change(
            params.forecast_bars, fill_method=None
        ).replace([np.inf, -np.inf], np.nan)
        out["mature_phase"] = out["phase"].shift(params.forecast_bars)
        out["seasonal_score"] = np.nan
        for phase in range(5):
            observed = (out["mature_phase"].eq(phase) &
                        out["mature_return"].notna())
            totals = out["mature_return"].where(observed, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            counts = observed.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            estimate = totals.div(counts.where(counts >= 8))
            out["phase_mean_" + str(phase)] = estimate
            out["seasonal_score"] = out["seasonal_score"].where(
                ~out["phase"].eq(phase), estimate
            )
        out["seasonal_threshold"] = out["seasonal_score"].abs().rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).quantile(params.threshold_quantile).shift(1)
        out["volume_baseline"] = volume.rolling(
            params.volume_bars, min_periods=params.volume_bars
        ).median().shift(1)
        out["relative_volume"] = volume.div(
            out["volume_baseline"].where(out["volume_baseline"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["daily_return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["volume_pressure"] = (
            out["daily_return"] * out["relative_volume"]
        ).rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure_threshold"] = out["volume_pressure"].abs().rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).quantile(params.threshold_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalVolumeParams,
    ) -> SignalFrame:
        score = indicators["seasonal_score"]
        pressure = indicators["volume_pressure"]
        ready = indicators[[
            "seasonal_score", "seasonal_threshold",
            "volume_pressure", "pressure_threshold",
        ]].notna().all(axis=1)
        eligible = (
            ready &
            score.abs().ge(indicators["seasonal_threshold"]) &
            pressure.abs().ge(indicators["pressure_threshold"])
        )
        long_entry = eligible & score.gt(0) & pressure.gt(0)
        short_entry = eligible & score.lt(0) & pressure.lt(0)
        candidates = np.select(
            [long_entry.to_numpy(), short_entry.to_numpy()],
            [1, -1], default=0
        ).astype(int)
        state = 0
        positions = np.zeros(len(data), dtype=int)
        for bar, candidate in enumerate(candidates):
            if state == 0:
                state = int(candidate)
            elif candidate == -state:
                state = 0
            positions[bar] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
