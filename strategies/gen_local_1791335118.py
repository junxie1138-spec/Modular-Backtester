from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    neighbors: int = 9
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791335118"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if not 3 <= params.neighbors <= 25 or params.hold_bars not in (3, 4, 5):
            raise ValueError("neighbors must be 3..25; hold_bars must be 3, 4, or 5")
        return 510

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        norm = returns.pow(2).rolling(5, min_periods=5).sum().pow(0.5)
        norm = norm.where(norm > 0)
        features = pd.DataFrame(
            {f"shape_{j}": returns.shift(j) / norm for j in range(5)},
            index=data.index,
        )
        columns = {"return": returns, "shape_norm": norm}
        columns.update({name: features[name] for name in features})
        distances, outcomes = [], []
        # Space analogs so their five-bar shapes and outcomes do not overlap.
        for lag in range(params.hold_bars + 5, 505, params.hold_bars + 5):
            distance = np.sum(
                (features.to_numpy() - features.shift(lag).to_numpy()) ** 2,
                axis=1,
            )
            outcome = (
                data["close"].shift(lag - params.hold_bars)
                / data["close"].shift(lag) - 1
            ).replace([np.inf, -np.inf], np.nan).to_numpy()
            columns[f"distance_{lag}"] = distance
            columns[f"outcome_{lag}"] = outcome
            distances.append(distance)
            outcomes.append(outcome)
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        valid = np.isfinite(distance_matrix) & np.isfinite(outcome_matrix)
        ranked_distance = np.where(valid, distance_matrix, np.inf)
        nearest = np.argsort(ranked_distance, axis=1, kind="stable")[:, :params.neighbors]
        selected_distance = np.take_along_axis(ranked_distance, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcome_matrix, nearest, axis=1)
        ready = np.isfinite(selected_distance).all(axis=1)
        columns["analog_mean"] = np.where(ready, selected_outcome.mean(axis=1), np.nan)
        columns["analog_win_rate"] = np.where(
            ready, (selected_outcome > 0).mean(axis=1), np.nan
        )
        columns["worst_distance"] = np.where(
            ready, selected_distance.max(axis=1), np.nan
        )
        return pd.DataFrame(columns, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entry = (
            (indicators["analog_mean"] > 0.002)
            & (indicators["analog_win_rate"] >= 2 / 3)
            & (indicators["worst_distance"] <= 0.6)
            & (np.arange(len(data)) >= warmup)
        ).fillna(False).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    raw[i] = 1
                else:
                    # Force a flat bar; entry evidence cannot postpone the exit.
                    exit_bar = -1
            elif entry[i]:
                raw[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=550, freq="B")
    close = 100 * np.exp(np.cumsum(0.001 + 0.01 * np.sin(np.arange(550))))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0}, index=index
    )
    params = AnalogParams()
    actual = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        actual.iloc[:530], strategy.indicators(data.iloc[:530], params)
    )
    forced = pd.DataFrame(
        {"analog_mean": 0.01, "analog_win_rate": 1.0, "worst_distance": 0.0},
        index=index,
    )
    for hold in (3, 4, 5):
        params = AnalogParams(hold_bars=hold)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        signals = strategy.generate_signals(data, forced, ctx, params).data
        first = strategy.warmup_bars(params) + 1
        assert (signals["signal"].iloc[:first] == 0).all()
        assert (signals["signal"].iloc[first:first + hold] == 1).all()
        assert signals["signal"].iloc[first + hold] == 0
        assert set(signals["signal"].unique()) <= {0, 1}
        assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
