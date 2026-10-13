from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    volatility_window: int = 6
    history_bars: int = 252
    neighbors: int = 7
    max_distance: float = 0.20
    min_edge: float = 0.0015
    min_win_rate: float = 0.60
    hold_bars: int = 2

    def __post_init__(self):
        if self.volatility_window < 2 or self.history_bars < self.neighbors or self.neighbors < 1:
            raise ValueError("Invalid volatility window or analog sample size")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        if not np.isfinite(self.max_distance) or self.max_distance <= 0:
            raise ValueError("max_distance must be finite and positive")
        if not np.isfinite(self.min_edge) or self.min_edge < 0:
            raise ValueError("min_edge must be finite and nonnegative")
        if not 0 < self.min_win_rate <= 1:
            raise ValueError("min_win_rate must lie in (0, 1]")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791201412"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        max_lag = params.hold_bars + params.volatility_window + 6 + params.history_bars - 1
        return int(max(200, params.volatility_window + 60) + max_lag)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["overnight_return"] = np.log(opening / close.shift(1))
        out["intraday_return"] = np.log(close / opening)
        out["overnight_std"] = out["overnight_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["intraday_std"] = out["intraday_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["total_std"] = np.sqrt(out["overnight_std"] ** 2 + out["intraday_std"] ** 2)
        out["baseline_std"] = out["total_std"].rolling(60, min_periods=60).mean()
        out["overnight_share"] = out["overnight_std"] ** 2 / out["total_std"].pow(2).replace(0, np.nan)
        out["share_lag3"] = out["overnight_share"].shift(3)
        out["share_lag6"] = out["overnight_share"].shift(6)
        out["relative_std"] = 0.5 * np.log(
            out["total_std"].replace(0, np.nan) / out["baseline_std"].replace(0, np.nan)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["share_lag6", "share_lag3", "overnight_share", "relative_std"]].to_numpy()
        prices = close.to_numpy()
        regime = (close > out["ma200"]).fillna(False).to_numpy()
        n = len(data)
        distances = np.full((n, params.history_bars), np.inf)
        outcomes = np.full_like(distances, np.nan)
        first_lag = params.hold_bars + params.volatility_window + 6
        # ponytail: O(bars * history) storage; process row blocks if intraday histories grow.
        for column, lag in enumerate(range(first_lag, first_lag + params.history_bars)):
            if lag >= n:
                break
            delta = features[lag:] - features[:-lag]
            distance = np.sqrt(np.mean(delta * delta, axis=1))
            realized = prices[params.hold_bars:n - lag + params.hold_bars] / prices[:n - lag] - 1.0
            valid = np.isfinite(distance) & np.isfinite(realized) & regime[:-lag]
            distances[lag:, column] = np.where(valid, distance, np.inf)
            outcomes[lag:, column] = np.where(valid, realized, np.nan)
        nearest = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distance = np.take_along_axis(distances, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcomes, nearest, axis=1)
        usable = np.isfinite(selected_distance) & (selected_distance <= params.max_distance)
        count = usable.sum(axis=1)
        denominator = np.maximum(count, 1)
        out["match_count"] = count
        out["expected_return"] = np.where(usable, selected_outcome, 0.0).sum(axis=1) / denominator
        out["win_rate"] = (usable & (selected_outcome > 0)).sum(axis=1) / denominator
        out["match_distance"] = np.max(selected_distance, axis=1)
        out.loc[count < params.neighbors, ["expected_return", "win_rate", "match_distance"]] = np.nan
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        eligible = (
            (data["close"] > indicators["ma200"])
            & (indicators["match_count"] >= params.neighbors)
            & (indicators["match_distance"] <= params.max_distance)
            & (indicators["expected_return"] > params.min_edge)
            & (indicators["win_rate"] >= params.min_win_rate)
            & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                signal[i] = 0
            elif eligible[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AnalogParams()
    strategy = GeneratedStrategy()
    start = strategy.warmup_bars(params) - 1
    index = pd.date_range("2000-01-01", periods=start + 8)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({
        "ma200": 99.0, "match_count": params.neighbors,
        "match_distance": 0.1, "expected_return": 0.01, "win_rate": 1.0,
    }, index=index)
    indicators.loc[index[start + 1]:, "expected_return"] = -0.01
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[:start + 1].eq(0).all()
    assert result["signal"].iloc[start + 1:start + 3].tolist() == [1, 1]
    assert result["signal"].iloc[start + 3:].eq(0).all()
