from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_bars: int = 20
    history_bars: int = 126
    neighbors: int = 7
    score_threshold: float = 0.25
    max_distance: float = 1.0
    hold_bars: int = 2

    def __post_init__(self):
        if not (self.ma_bars >= 2 and self.history_bars >= self.neighbors >= 1):
            raise ValueError("Require ma_bars >= 2 and history_bars >= neighbors >= 1")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not (np.isfinite(self.score_threshold) and self.score_threshold > 0
                and np.isfinite(self.max_distance) and self.max_distance > 0):
            raise ValueError("Thresholds must be finite and positive")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791275605"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params):
        return int(params.ma_bars + params.history_bars + params.hold_bars + 5)

    def indicators(self, data, params):
        close = data["close"].astype(float)
        opening = data["open"].astype(float).where(data["open"] > 0)
        mean = close.rolling(params.ma_bars).mean()
        spread = close.rolling(params.ma_bars).std(ddof=0).replace(0, np.nan)
        z = ((close - mean) / spread).replace([np.inf, -np.inf], np.nan)
        vol = close.pct_change(fill_method=None).rolling(params.ma_bars).std(ddof=0)
        vol = vol.where(vol > 0)
        out = pd.DataFrame({"ma": mean, "spread": spread,
                            "z": z, "z_previous": z.shift(1),
                            "z_previous2": z.shift(2), "vol": vol}, index=data.index)
        distances, outcomes = [], []
        # Completed analog outcomes precede every bar of the current three-bar shape.
        for lag in range(params.hold_bars + 4,
                         params.hold_bars + 4 + params.history_bars):
            distance = ((z - z.shift(lag)).pow(2)
                        + (z.shift(1) - z.shift(lag + 1)).pow(2)
                        + (z.shift(2) - z.shift(lag + 2)).pow(2)).div(3).pow(0.5)
            outcome = (opening.shift(lag - params.hold_bars - 1)
                       / opening.shift(lag - 1) - 1)
            outcome = outcome / (vol.shift(lag) * np.sqrt(params.hold_bars))
            distances.append(distance.to_numpy())
            outcomes.append(outcome.to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        valid = np.isfinite(distance_matrix) & np.isfinite(outcome_matrix)
        distance_matrix = np.where(valid, distance_matrix, np.inf)
        selected = np.argpartition(distance_matrix, params.neighbors - 1, axis=1)
        selected = selected[:, :params.neighbors]
        matched_distance = np.take_along_axis(distance_matrix, selected, axis=1)
        matched_outcome = np.take_along_axis(outcome_matrix, selected, axis=1)
        weights = 1.0 / (0.05 + matched_distance)
        weighted_sum = (weights * np.where(np.isfinite(matched_outcome),
                                            matched_outcome, 0.0)).sum(axis=1)
        total_weight = weights.sum(axis=1)
        forecast = np.divide(weighted_sum, total_weight,
                             out=np.full(len(data), np.nan), where=total_weight > 0)
        out["forecast"] = forecast
        out["match_distance"] = np.max(matched_distance, axis=1)
        out["ready"] = ((np.arange(len(data)) >= self.warmup_bars(params) - 1)
                        & np.isfinite(matched_distance).all(axis=1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        usable = (indicators["ready"].fillna(False)
                  & indicators["match_distance"].le(params.max_distance)
                  & indicators["forecast"].notna())
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[usable & indicators["forecast"].ge(params.score_threshold)] = 1
        candidate.loc[usable & indicators["forecast"].le(-params.score_threshold)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        for i, direction in enumerate(confirmed):
            if position:
                if i - entry_bar == params.hold_bars:
                    position = 0
                # No reversal or re-entry on the scheduled exit bar.
            elif direction:
                position, entry_bar = int(direction), i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=8))
    indicators = pd.DataFrame({"ready": True, "match_distance": 0.0,
                               "forecast": 1.0}, index=data.index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 1, 1, 0]
    indicators["forecast"] = -1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
