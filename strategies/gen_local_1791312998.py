from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    correlation_floor: float = 0.15
    forecast_threshold: float = 0.35
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791312998"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        price = np.log(data["close"].where(data["close"] > 0))
        returns = price.diff()
        volatility = returns.rolling(8, min_periods=8).std(ddof=0)
        correlation = returns.rolling(8, min_periods=8).corr(returns.shift(1))
        weighted_return = pd.Series(0.0, index=data.index)
        weight_sum = pd.Series(0.0, index=data.index)
        degree = pd.Series(0, index=data.index, dtype=int)
        valid = price.rolling(9, min_periods=9).count().eq(9)
        for lag in range(1, 8):
            visible = price.notna() & price.shift(lag).notna()
            if lag > 1:
                barrier = price.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
                endpoint_floor = pd.concat([price, price.shift(lag)], axis=1).min(axis=1)
                visible &= endpoint_floor.gt(barrier)
            usable = visible & returns.shift(lag).notna()
            weight = usable.astype(float) / lag
            weighted_return += returns.shift(lag).fillna(0.0) * weight
            weight_sum += weight
            degree += visible.astype(int)
        neighbor_return = weighted_return.div(weight_sum.where(weight_sum > 0))
        score = neighbor_return.div(volatility.where(volatility > 0)) * np.sign(correlation)
        ready = valid & correlation.notna() & volatility.gt(0)
        out["log_close"] = price
        out["return"] = returns
        out["volatility"] = volatility
        out["autocorrelation"] = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["visibility_degree"] = degree.where(ready, 0).astype(int)
        out["neighbor_return"] = neighbor_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["forecast_score"] = score.where(ready, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["ready"] = ready
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if not 0.0 <= params.correlation_floor <= 1.0:
            raise ValueError("correlation_floor must be between zero and one")
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError("forecast_threshold must be finite and positive")
        eligible = (
            indicators["ready"].fillna(False)
            & indicators["visibility_degree"].ge(3)
            & indicators["autocorrelation"].abs().ge(params.correlation_floor)
        )
        score = indicators["forecast_score"].fillna(0.0)
        entries = np.select(
            [eligible & score.ge(params.forecast_threshold), eligible & score.le(-params.forecast_threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar == params.hold_bars:
                    position = 0
                    continue
                signals[i] = position
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
                signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilityParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=40))
    for direction in (1, -1):
        indicators = pd.DataFrame({
            "ready": True,
            "visibility_degree": 3,
            "autocorrelation": 0.5,
            "forecast_score": float(direction),
        }, index=data.index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        expected = [0] + [direction] * params.hold_bars + [0]
        assert result.data["signal"].iloc[:len(expected)].tolist() == expected
        assert result.data["signal"].iloc[20] == direction
        assert result.data["signal"].iloc[38] == 0
        assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
