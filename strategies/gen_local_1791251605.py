from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    block_bars: int = 5
    rank_window: int = 126
    analog_history: int = 252
    neighbors: int = 12
    forecast_window: int = 126
    pressure_bars: int = 5
    entry_percentile: float = 0.80
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791251605"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(
            3 * params.block_bars + params.rank_window
            + params.hold_bars + params.analog_history
            + params.forecast_window,
            params.pressure_bars + params.rank_window,
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if min(params.block_bars, params.rank_window, params.analog_history,
               params.neighbors, params.forecast_window, params.pressure_bars,
               params.hold_bars) < 1:
            raise ValueError("Lookbacks, neighbors, and holding period must be positive")
        if params.neighbors > params.analog_history:
            raise ValueError("neighbors must not exceed analog_history")
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between 0.5 and 1")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["block_return"] = close.pct_change(params.block_bars, fill_method=None)
        out["return_rank"] = out["block_return"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["prior_rank"] = out["return_rank"].shift(params.block_bars)
        out["older_rank"] = out["return_rank"].shift(2 * params.block_bars)
        features = out[["return_rank", "prior_rank", "older_rank"]]

        out["matured_return"] = close.pct_change(params.hold_bars, fill_method=None)
        distances = np.full((len(data), params.analog_history), np.inf)
        outcomes = np.full_like(distances, np.nan)
        for column in range(params.analog_history):
            lag = params.hold_bars + 1 + column
            difference = features.to_numpy() - features.shift(lag).to_numpy()
            distance = np.sum(difference * difference, axis=1)
            outcome = out["matured_return"].shift(lag - params.hold_bars).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[:, column] = np.where(valid, distance, np.inf)
            outcomes[:, column] = np.where(valid, outcome, np.nan)

        nearest = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distances, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        out["analog_forecast"] = np.where(
            complete, np.mean(selected_outcomes, axis=1), np.nan
        )
        out["forecast_rank"] = out["analog_forecast"].rolling(
            params.forecast_window, min_periods=params.forecast_window
        ).rank(pct=True)

        candle_range = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["candle_pressure"] = (
            (data["close"] - data["open"]) / candle_range
        ).clip(-1.0, 1.0).where(candle_range.notna(), 0.0)
        volume = data["volume"].clip(lower=0)
        out["weighted_pressure"] = out["candle_pressure"] * volume
        out["pressure_volume"] = volume.rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum()
        out["pressure"] = out["weighted_pressure"].rolling(
            params.pressure_bars, min_periods=params.pressure_bars
        ).sum() / out["pressure_volume"].where(out["pressure_volume"] > 0)
        out["pressure_rank"] = out["pressure"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _fixed_hold(entries: np.ndarray, hold_bars: int) -> np.ndarray:
        signals = np.zeros(len(entries), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0:
                if i - entry_bar >= hold_bars:
                    position = 0
                signals[i] = position
            elif candidate != 0:
                position = int(candidate)
                entry_bar = i
                signals[i] = position
        return signals

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        upper = params.entry_percentile
        lower = 1.0 - upper
        long_entry = (
            (indicators["analog_forecast"] > 0)
            & (indicators["forecast_rank"] >= upper)
            & (indicators["pressure"] > 0)
            & (indicators["pressure_rank"] >= upper)
        )
        short_entry = (
            (indicators["analog_forecast"] < 0)
            & (indicators["forecast_rank"] <= lower)
            & (indicators["pressure"] < 0)
            & (indicators["pressure_rank"] <= lower)
        )
        entries = np.select(
            [long_entry.fillna(False), short_entry.fillna(False)],
            [1, -1], default=0,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._fixed_hold(entries, params.hold_bars)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    entries = np.array([1, -1, -1, 1, -1, 1, 1, -1])
    decisions = GeneratedStrategy._fixed_hold(entries, 3)
    fills = pd.Series(decisions).shift(1).fillna(0).astype(int)
    assert fills.tolist() == [0, 1, 1, 1, 0, -1, -1, -1]
