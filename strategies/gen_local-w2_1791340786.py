from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    calibration_window: int = 126
    hub_percentile: float = 0.85
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791340786"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(params.graph_window + params.calibration_window + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.calibration_window < 2:
            raise ValueError("Lookback windows are too short")
        if not 0.0 < params.hub_percentile < 1.0:
            raise ValueError("hub_percentile must be between zero and one")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be three, four, or five")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        window = params.graph_window
        degree = pd.Series(0.0, index=data.index)
        blocker = pd.Series(np.inf, index=data.index)
        # Inverted horizontal visibility: intervening closes must exceed both endpoints.
        for lag in range(1, window):
            if lag > 1:
                blocker = pd.Series(
                    np.minimum(blocker.to_numpy(), close.shift(lag - 1).to_numpy()),
                    index=data.index,
                )
            visible = (close < blocker) & (close.shift(lag) < blocker)
            degree += visible.astype(float)
        degree = degree.where(close.rolling(window, min_periods=window).count().eq(window))
        floor = close.rolling(window, min_periods=window).min()
        ceiling = close.rolling(window, min_periods=window).max()
        position = (close - floor) / (ceiling - floor).replace(0.0, np.nan)
        hub_threshold = degree.shift(1).rolling(
            params.calibration_window, min_periods=params.calibration_window
        ).quantile(params.hub_percentile)
        position_threshold = position.shift(1).rolling(
            params.calibration_window, min_periods=params.calibration_window
        ).quantile(0.5)
        rebound = close.gt(close.shift(1))
        eligible = (
            degree.shift(1).gt(hub_threshold.shift(1))
            & degree.lt(degree.shift(1))
            & position.lt(position_threshold)
            & rebound
        ).fillna(False)
        return pd.DataFrame({
            "degree": degree,
            "range_floor": floor,
            "range_ceiling": ceiling,
            "position": position,
            "hub_threshold": hub_threshold,
            "position_threshold": position_threshold,
            "rebound": rebound,
            "eligible": eligible,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be three, four, or five")
        entries = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force an exit even if another entry condition is present.
                    exit_bar = -1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"close": np.ones(12)}, index=index)
    indicators = pd.DataFrame({"eligible": True}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert result["size"].gt(0).all()
    prices = pd.DataFrame({"close": [12., 10., 11., 9., 8., 10., 7., 11.]})
    graph_params = VisibilityParams(graph_window=4, calibration_window=2)
    degree = strategy.indicators(prices, graph_params)["degree"]
    values = prices["close"].to_numpy()
    for t in range(3, len(values)):
        expected = sum(
            all(values[k] > max(values[t], values[j]) for k in range(j + 1, t))
            for j in range(t - 3, t)
        )
        assert degree.iloc[t] == expected


if __name__ == "__main__":
    _self_check()
