from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    max_distance: float = 0.35
    min_followthrough: float = 0.25
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1790866376"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        # Three returns require four closes; the oldest analog is six bars back.
        return 10

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        r = out["return"]
        variation = r.abs().rolling(3, min_periods=3).sum().replace(0, np.nan)
        out["variation"] = variation
        out["displacement"] = r.rolling(3, min_periods=3).sum() / variation
        # Antisymmetric level-two signature of (normalized time, cumulative return).
        # Equal time steps give area = (newest return - oldest return) / 3.
        out["area"] = (r - r.shift(2)) / (3.0 * variation)
        distances = []
        outcomes = []
        for lag in (4, 5, 6):
            # The analog's next return ends before the current three-return path.
            out[f"analog_area_{lag}"] = out["area"].shift(lag)
            out[f"analog_displacement_{lag}"] = out["displacement"].shift(lag)
            distance = np.sqrt(
                (out["area"] - out[f"analog_area_{lag}"]) ** 2
                + (out["displacement"] - out[f"analog_displacement_{lag}"]) ** 2
            )
            outcome = r.shift(lag - 1) / (variation.shift(lag) / 3.0)
            valid = distance.notna() & outcome.notna()
            out[f"distance_{lag}"] = distance.where(valid, 10.0)
            out[f"outcome_{lag}"] = outcome.where(valid, 0.0)
            distances.append(out[f"distance_{lag}"].to_numpy())
            outcomes.append(out[f"outcome_{lag}"].to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = distance_matrix.argmin(axis=1)
        rows = np.arange(len(data))
        out["distance"] = distance_matrix[rows, nearest]
        out["forecast"] = outcome_matrix[rows, nearest]
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        if (
            not isinstance(params.hold_bars, int)
            or not 15 <= params.hold_bars <= 20
            or not np.isfinite(params.max_distance)
            or not 0 < params.max_distance <= 2
            or not np.isfinite(params.min_followthrough)
            or params.min_followthrough < 0
        ):
            raise ValueError("Require 15-20 holding bars and finite nonnegative thresholds.")
        entry = (
            (indicators["displacement"] < 0)
            & (indicators["area"] > 0)
            & (indicators["distance"] <= params.max_distance)
            & (indicators["forecast"] >= params.min_followthrough)
        ).fillna(False).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entry[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShapeParams()
    index = pd.date_range("2020-01-01", periods=65, freq="B")
    close = 100.0 * np.exp(np.cumsum(0.01 * np.sin(np.arange(65))))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0}, index=index
    )
    features = strategy.indicators(data, params)
    assert np.isfinite(features.to_numpy()).all()
    pd.testing.assert_frame_equal(
        features.iloc[:30], strategy.indicators(data.iloc[:30], params)
    )
    forced = pd.DataFrame(
        {"displacement": -0.5, "area": 0.1, "distance": 0.0, "forecast": 1.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", 10)
    signals = strategy.generate_signals(data, forced, ctx, params).data
    expected = np.zeros(65, dtype=int)
    for start in (10, 29, 48):
        expected[start:start + params.hold_bars] = 1
    assert np.array_equal(signals["signal"].to_numpy(), expected)
    assert (signals["size"] > 0).all()
    flat = data.assign(close=100.0)
    flat_features = strategy.indicators(flat, params)
    assert np.isfinite(flat_features.to_numpy()).all()
    assert not strategy.generate_signals(flat, flat_features, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
