from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveComplexityParams:
    ordinal_lag: int = 5
    pattern_window: int = 20
    smoothing_window: int = 10
    calibration_window: int = 126
    complexity_quantile: float = 0.60
    breakout_window: int = 20
    breakout_quantile: float = 0.70
    pressure_window: int = 20
    pressure_quantile: float = 0.60


class GeneratedStrategy(BaseStrategy[WaveComplexityParams]):
    strategy_id = "gen_local-w2_1790995636"

    @classmethod
    def params_type(cls) -> type[WaveComplexityParams]:
        return WaveComplexityParams

    def warmup_bars(self, params: WaveComplexityParams) -> int:
        ordinal = (
            2 * params.ordinal_lag + params.pattern_window
            + params.smoothing_window + params.calibration_window
        )
        breakout = (
            params.breakout_window + params.pressure_window
            + params.calibration_window
        )
        return int(max(ordinal, breakout))

    def indicators(self, data: pd.DataFrame, params: WaveComplexityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ordinal_old"] = close.shift(2 * params.ordinal_lag)
        out["ordinal_middle"] = close.shift(params.ordinal_lag)
        out["ordinal_new"] = close
        a = out["ordinal_old"]
        b = out["ordinal_middle"]
        c = out["ordinal_new"]
        valid = a.notna() & b.notna() & c.notna()
        # Strict comparisons resolve ties by chronological order.
        out["ordinal_code"] = (
            4 * (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + (b > c).astype(int)
        ).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            occurrence = f"occurrence_{code}"
            probability = f"probability_{code}"
            out[occurrence] = (out["ordinal_code"] == code).astype(float).where(valid)
            out[probability] = out[occurrence].rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probability_columns.append(probability)
        probabilities = out[probability_columns]
        safe_probabilities = probabilities.where(probabilities > 0.0, 1.0)
        out["entropy"] = -(
            probabilities * np.log(safe_probabilities)
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = (
            (probabilities - 1.0 / 6.0) ** 2
        ).sum(axis=1, min_count=6) * 6.0 / 5.0
        out["raw_complexity"] = out["entropy"] * out["disequilibrium"]
        out["complexity"] = out["raw_complexity"].rolling(
            params.smoothing_window, min_periods=params.smoothing_window
        ).mean()
        out["complexity_threshold"] = out["complexity"].shift(1).rolling(
            params.calibration_window, min_periods=params.calibration_window
        ).quantile(params.complexity_quantile)
        out["breakout_level"] = close.shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).quantile(params.breakout_quantile)
        out["breakout_event"] = (
            close > out["breakout_level"]
        ).astype(float).where(close.notna() & out["breakout_level"].notna())
        out["breakout_pressure"] = out["breakout_event"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).mean()
        out["pressure_threshold"] = out["breakout_pressure"].shift(1).rolling(
            params.calibration_window, min_periods=params.calibration_window
        ).quantile(params.pressure_quantile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveComplexityParams,
    ) -> SignalFrame:
        required = [
            "complexity", "complexity_threshold",
            "breakout_pressure", "pressure_threshold",
        ]
        ready = indicators[required].notna().all(axis=1)
        entry_condition = (
            ready
            & (indicators["complexity"] > indicators["complexity_threshold"])
            & (indicators["breakout_pressure"] > indicators["pressure_threshold"])
        )
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; its reversal is the sole exit.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
