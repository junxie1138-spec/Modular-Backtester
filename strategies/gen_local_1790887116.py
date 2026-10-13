from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    pattern_window: int = 20
    history_window: int = 252
    forecast_horizon: int = 8
    threshold_window: int = 126
    neighbor_percentile: float = 0.10
    entry_percentile: float = 0.70

    def __post_init__(self):
        if self.pattern_window < 3 or self.forecast_horizon < 1 or self.threshold_window < 2:
            raise ValueError("Lookbacks must be positive and pattern_window at least three.")
        if self.history_window < self.pattern_window + self.forecast_horizon + 18:
            raise ValueError("history_window must leave at least eight eligible analogs.")
        if not 0 < self.neighbor_percentile < 1 or not 0 < self.entry_percentile < 1:
            raise ValueError("Percentiles must lie strictly between zero and one.")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790887116"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_window + params.pattern_window + params.threshold_window + 13)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        out["return"] = np.log(close).diff()
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Ties are ordered by time: the older observation ranks first.
        code = (a <= b).astype(int) + 2 * (a <= c).astype(int) + 4 * (b <= c).astype(int)
        out["pattern"] = code.where(valid)
        probability_columns = []
        for pattern in (0, 1, 3, 4, 6, 7):
            name = "p" + str(pattern)
            probability_columns.append(name)
            out[name] = out["pattern"].eq(pattern).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
        probabilities = out[probability_columns]
        out["complexity"] = -(probabilities * np.log(probabilities.replace(0.0, 1.0))).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        out["complexity_5"] = out["complexity"].shift(5)
        out["complexity_10"] = out["complexity"].shift(10)
        out["ordinal_tension"] = out["p7"] - out["p0"]
        features = out[["complexity", "complexity_5", "complexity_10", "ordinal_tension"]]
        distances = {}
        outcomes = {}
        first_lag = params.pattern_window + params.forecast_horizon + 10
        for lag in range(first_lag, params.history_window + 1):
            distances[lag] = (features - features.shift(lag)).pow(2).sum(axis=1, min_count=4)
            # Every endpoint precedes the entire current signature.
            outcomes[lag] = close.shift(lag - params.forecast_horizon) / close.shift(lag) - 1.0
        distance_frame = pd.DataFrame(distances, index=data.index)
        outcome_frame = pd.DataFrame(outcomes, index=data.index)
        out["neighbor_cutoff"] = distance_frame.quantile(params.neighbor_percentile, axis=1)
        selected = distance_frame.le(out["neighbor_cutoff"], axis=0) & outcome_frame.notna()
        out["analog_count"] = selected.sum(axis=1).astype(float)
        forecast = outcome_frame.where(selected).sum(axis=1, min_count=1).div(
            out["analog_count"].replace(0.0, np.nan)
        )
        out["forecast"] = forecast.where(distance_frame.notna().all(axis=1))
        out["entry_cutoff"] = out["forecast"].abs().shift(1).rolling(
            params.threshold_window, min_periods=params.threshold_window
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        forecast = indicators["forecast"]
        cutoff = indicators["entry_cutoff"]
        qualified = forecast.notna() & cutoff.notna() & (forecast.abs() > cutoff)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[qualified & (forecast > 0)] = 1
        candidate.loc[qualified & (forecast < 0)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Reversal closes to cash; the next qualifying bar can reenter.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2000-01-01", periods=700, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(17).normal(0.0002, 0.012, len(index))))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0},
        index=index
    )
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:600], params)
    pd.testing.assert_frame_equal(full.iloc[:600], prefix)
    signal = strategy.generate_signals(data, full, None, params).data
    pd.testing.assert_frame_equal(
        signal.iloc[:600], strategy.generate_signals(data.iloc[:600], prefix, None, params).data
    )
    assert signal["signal"].isin([-1, 0, 1]).all() and (signal["size"] > 0).all()
    assert full["entry_cutoff"].notna().any()
    probe = pd.DataFrame(
        {"forecast": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0], "entry_cutoff": 0.5},
        index=index[:7]
    )
    result = strategy.generate_signals(data.iloc[:7], probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
