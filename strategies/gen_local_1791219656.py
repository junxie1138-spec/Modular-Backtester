from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    seasonal_threshold: float = 1.0
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791219656"

    @classmethod
    def params_type(cls) -> type[SeasonalParams]:
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return 820

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0))
        weights = np.ones(64, dtype=float)
        for lag in range(1, 64):
            weights[lag] = -weights[lag - 1] * (0.4 - lag + 1) / lag
        fractional_close = log_close * weights[0]
        for lag in range(1, 64):
            fractional_close = fractional_close + log_close.shift(lag) * weights[lag]
        out["fractional_close"] = fractional_close
        out["innovation"] = fractional_close.diff()

        year_days = np.where(data.index.is_leap_year, 366.0, 365.0)
        phase = pd.Series(
            2.0 * np.pi * (data.index.dayofyear.to_numpy() - 1) / year_days,
            index=data.index,
        )
        out["phase"] = phase
        out["cos_phase"] = np.cos(phase)
        out["sin_phase"] = np.sin(phase)
        centered = out["innovation"] - out["innovation"].rolling(756, min_periods=756).mean()
        # Subtract the rolling mean after projection to preserve the full sample.
        mean_innovation = out["innovation"].rolling(756, min_periods=756).mean()
        mean_cos = out["cos_phase"].rolling(756, min_periods=756).mean()
        mean_sin = out["sin_phase"].rolling(756, min_periods=756).mean()
        out["centered_innovation"] = centered
        out["cos_coefficient"] = 2.0 * (
            (out["innovation"] * out["cos_phase"]).rolling(756, min_periods=756).mean()
            - mean_innovation * mean_cos
        )
        out["sin_coefficient"] = 2.0 * (
            (out["innovation"] * out["sin_phase"]).rolling(756, min_periods=756).mean()
            - mean_innovation * mean_sin
        )
        out["innovation_std"] = out["innovation"].rolling(756, min_periods=756).std()
        # Eighteen sessions correspond approximately to twenty-five calendar days.
        half_span = np.pi * 25.0 / 365.2425
        midpoint = phase + half_span
        out["forecast"] = (np.sin(half_span) / half_span) * (
            out["cos_coefficient"] * np.cos(midpoint)
            + out["sin_coefficient"] * np.sin(midpoint)
        )
        scale = out["innovation_std"] * np.sqrt(2.0 / 756.0)
        out["score"] = (out["forecast"] / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)

        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalParams,
    ) -> SignalFrame:
        if not np.isfinite(params.seasonal_threshold) or params.seasonal_threshold <= 0:
            raise ValueError("seasonal_threshold must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")

        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        side = 0
        pending_entry = False
        entry_price = np.nan
        entry_atr = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                entry_price = opens[i]
                entry_bar = i
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0

            if side != 0:
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                stopped = valid_close and (
                    (side == 1 and closes[i] < entry_price - params.stop_atr * entry_atr)
                    or (side == -1 and closes[i] > entry_price + params.stop_atr * entry_atr)
                )
                if stopped or i - entry_bar >= 17 or not valid_close:
                    side = 0
                decisions[i] = side
                continue

            if (
                i >= self.warmup_bars(params) - 1
                and np.isfinite(scores[i])
                and np.isfinite(atrs[i])
                and atrs[i] > 0
                and np.isfinite(closes[i])
                and closes[i] > 0
                and abs(scores[i]) >= params.seasonal_threshold
            ):
                side = 1 if scores[i] > 0 else -1
                entry_atr = atrs[i]
                pending_entry = True
            decisions[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
