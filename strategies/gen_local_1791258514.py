from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory: int = 16
    fractional_order: float = 0.6
    threshold_window: int = 63
    percentile: float = 0.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791258514"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.memory + params.threshold_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.memory >= 2 and params.threshold_window >= 5
                and 0.0 < params.fractional_order < 1.0
                and 0.5 < params.percentile < 1.0):
            raise ValueError("Invalid fractional-memory or percentile parameters.")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.memory, dtype=float)
        for k in range(1, params.memory):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: finite fractional memory; increase memory if truncation matters.
        fractional_close = close.copy()
        for k in range(1, params.memory):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["innovation"] = fractional_close.diff()
        history = out["innovation"].shift(1).rolling(
            params.threshold_window, min_periods=params.threshold_window
        )
        out["upper"] = history.quantile(params.percentile).clip(lower=0.0)
        out["lower"] = history.quantile(1.0 - params.percentile).clip(upper=0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        valid = indicators[["innovation", "upper", "lower"]].notna().all(axis=1)
        long_entry = indicators["innovation"].gt(indicators["upper"])
        short_entry = indicators["innovation"].lt(indicators["lower"])
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (ready, go_long, go_short) in enumerate(
            zip(valid.to_numpy(), long_entry.to_numpy(), short_entry.to_numpy())
        ):
            if ready:
                if position == 1:
                    if not go_long:
                        position = 0
                elif position == -1:
                    if not go_short:
                        position = 0
                elif go_long:
                    position = 1
                elif go_short:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    rng = np.random.default_rng(17)
    close = 100.0 + np.cumsum(rng.normal(size=240))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(close), freq="B")
    )
    features = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    prefix = data.iloc[:180]
    prefix_features = GeneratedStrategy.indicators(prefix, params)
    prefix_result = GeneratedStrategy.generate_signals(
        prefix, prefix_features, None, params
    ).data
    assert result.iloc[:180].equals(prefix_result)
    assert set(result["signal"].unique()) == {-1, 0, 1}
    assert result["size"].gt(0).all()
    assert result.index.equals(data.index)
    assert result["signal"].iloc[:GeneratedStrategy.warmup_bars(params)].eq(0).all()
    forced = pd.DataFrame(
        {"innovation": [2.0, 2.0, -2.0, -2.0, 0.0],
         "upper": 1.0, "lower": -1.0}, index=data.index[:5]
    )
    checked = GeneratedStrategy.generate_signals(
        data.iloc[:5], forced, None, params
    ).data
    assert checked["signal"].tolist() == [0, 1, 1, 0, -1]
