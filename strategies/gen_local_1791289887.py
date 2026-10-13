from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 3
    phase_bars: int = 3
    scale_bars: int = 20
    history_bars: int = 126
    neighbors: int = 9
    forecast_bars: int = 8
    percentile_bars: int = 126
    entry_percentile: float = 0.75

    def __post_init__(self):
        lengths = (self.roc_bars, self.phase_bars, self.scale_bars,
                   self.history_bars, self.neighbors, self.forecast_bars,
                   self.percentile_bars)
        if any(x < 1 for x in lengths):
            raise ValueError("Lookbacks and neighbor count must be positive.")
        if self.scale_bars < 2 or self.neighbors > self.history_bars:
            raise ValueError("Require scale_bars >= 2 and neighbors <= history_bars.")
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one.")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791289887"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        first_lag = params.forecast_bars + 3 * params.phase_bars + 2 * params.roc_bars
        last_lag = first_lag + params.history_bars - 1
        return int(2 * params.roc_bars + params.scale_bars
                   + 3 * params.phase_bars + last_lag + params.percentile_bars)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["roc"] = close.pct_change(p.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff(p.roc_bars)
        out["acceleration_scale"] = (
            out["acceleration"].rolling(p.scale_bars, min_periods=p.scale_bars)
            .std(ddof=0).shift(1).replace(0.0, np.nan)
        )
        out["scaled_acceleration"] = (
            out["acceleration"] / out["acceleration_scale"]
        ).replace([np.inf, -np.inf], np.nan)
        out["phase_recent"] = out["scaled_acceleration"].rolling(
            p.phase_bars, min_periods=p.phase_bars
        ).mean()
        out["phase_middle"] = out["phase_recent"].shift(p.phase_bars)
        out["phase_old"] = out["phase_recent"].shift(2 * p.phase_bars)
        out["realized_return"] = close.pct_change(p.forecast_bars, fill_method=None)
        features = out[["phase_old", "phase_middle", "phase_recent"]]
        current = features.to_numpy(dtype=float)
        n = len(data)
        distances = np.full((n, p.history_bars), np.inf)
        outcomes = np.full((n, p.history_bars), np.nan)
        first_lag = p.forecast_bars + 3 * p.phase_bars + 2 * p.roc_bars
        # Each analog's outcome ends before the current acceleration shape starts.
        for j in range(p.history_bars):
            lag = first_lag + j
            historical = features.shift(lag).to_numpy(dtype=float)
            outcome = out["realized_return"].shift(lag - p.forecast_bars).to_numpy()
            valid = (np.isfinite(current).all(axis=1)
                     & np.isfinite(historical).all(axis=1)
                     & np.isfinite(outcome))
            distance = np.sum((current - historical) ** 2, axis=1)
            distances[:, j] = np.where(valid, distance, np.inf)
            outcomes[:, j] = outcome
        # ponytail: exhaustive O(bars * history_bars) search; index shapes if the library grows substantially.
        nearest = np.argpartition(distances, p.neighbors - 1, axis=1)[:, :p.neighbors]
        chosen_distances = np.take_along_axis(distances, nearest, axis=1)
        chosen_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        ready = np.isfinite(distances).all(axis=1)
        out["analog_distance"] = np.where(ready, chosen_distances.mean(axis=1), np.nan)
        out["forecast"] = np.where(ready, chosen_outcomes.mean(axis=1), np.nan)
        out["entry_threshold"] = (
            out["forecast"].abs().shift(1)
            .rolling(p.percentile_bars, min_periods=p.percentile_bars)
            .quantile(p.entry_percentile)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        forecast = indicators["forecast"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        valid = np.isfinite(forecast) & np.isfinite(threshold)
        entries = np.where(valid & (forecast > threshold), 1,
                           np.where(valid & (forecast < -threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"forecast": [0.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0],
         "entry_threshold": 1.0}, index=index
    )
    result = strategy.generate_signals(data, indicators, None, AnalogParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
