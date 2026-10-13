from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarComplexityParams:
    pattern_window: int = 20
    percentile_window: int = 252
    complexity_quantile: float = 0.75
    season_window: int = 504
    min_season_samples: int = 40
    atr_window: int = 14
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CalendarComplexityParams]):
    strategy_id = "gen_local-w2_1790987840"

    @classmethod
    def params_type(cls):
        return CalendarComplexityParams

    def warmup_bars(self, params: CalendarComplexityParams) -> int:
        return int(max(
            params.season_window + params.hold_bars,
            params.pattern_window + params.percentile_window + 3,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: CalendarComplexityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None)
        out["lag1_return"] = out["return"].shift(1)
        out["lag2_return"] = out["return"].shift(2)
        a, b, c = out["lag2_return"], out["lag1_return"], out["return"]
        valid = pd.concat([a, b, c], axis=1).notna().all(axis=1)
        out["ordinal_code"] = (
            (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + 4 * (b > c).astype(int)
        ).where(valid)
        # Strict comparisons give stable chronological ordering for ties.
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = "pattern_prob_" + str(code)
            out[column] = out["ordinal_code"].eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
            axis=1, min_count=6
        )
        mixture = 0.5 * (probabilities + 1.0 / 6.0)
        mixture_entropy = -(mixture * np.log(mixture.clip(lower=1e-12))).sum(
            axis=1, min_count=6
        )
        out["permutation_entropy"] = entropy / np.log(6.0)
        out["ordinal_disequilibrium"] = (
            mixture_entropy - 0.5 * entropy - 0.5 * np.log(6.0)
        ).clip(lower=0.0) / np.log(2.0)
        out["complexity"] = out["permutation_entropy"] * out["ordinal_disequilibrium"]
        out["complexity_threshold"] = out["complexity"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.complexity_quantile).shift(1)

        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index, dtype=float)
        out["origin_weekday"] = out["weekday"].shift(params.hold_bars)
        # At t, this trade began at t-hold_bars+1 open and has now matured.
        out["matured_return"] = close / data["open"].shift(params.hold_bars - 1).replace(
            0.0, np.nan
        ) - 1.0
        conditions, forecasts = [], []
        for weekday in range(5):
            column = "weekday_edge_" + str(weekday)
            out[column] = out["matured_return"].where(
                out["origin_weekday"].eq(weekday)
            ).rolling(
                params.season_window, min_periods=params.min_season_samples
            ).mean()
            conditions.append(out["weekday"].eq(weekday))
            forecasts.append(out[column])
        out["seasonal_edge"] = np.select(conditions, forecasts, default=np.nan)

        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CalendarComplexityParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.95
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        complexity = indicators["complexity"].to_numpy(dtype=float)
        threshold = indicators["complexity_threshold"].to_numpy(dtype=float)
        edge = indicators["seasonal_edge"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        direction, age = 0, 0
        pending_entry = False
        entry = np.nan
        stop = -np.inf
        armed = False
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending_entry:
                # The previous close requested entry; this bar's open is now observable.
                entry = opens[i]
                pending_entry = False
                age = 0
                stop = -np.inf
                armed = False
                if not np.isfinite(entry) or entry <= 0.0:
                    direction = 0
                    continue

            if direction:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]):
                    direction = 0
                    continue
                signed_close = direction * close[i]
                signed_entry = direction * entry
                # Stops are close-confirmed; exits execute at the next open.
                if (armed and signed_close <= stop) or age >= params.hold_bars:
                    direction = 0
                    continue
                if signed_close - signed_entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-price stops only rise; raw short stops only fall.
                    stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
                signals[i] = direction
                continue

            if i < warmup - 1:
                continue
            if not all(np.isfinite(v) for v in (
                close[i], atr[i], complexity[i], threshold[i], edge[i]
            )):
                continue
            if close[i] <= 0.0 or atr[i] <= 0.0:
                continue
            if complexity[i] > threshold[i] and edge[i] != 0.0:
                direction = 1 if edge[i] > 0.0 else -1
                pending_entry = True
                signals[i] = direction

        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
