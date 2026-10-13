from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    lookback: int = 12
    min_streak: int = 3
    min_repair: float = 0.20
    max_repair: float = 0.65


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = "gen_local_1790911466"

    @classmethod
    def params_type(cls) -> type[RepairParams]:
        return RepairParams

    def warmup_bars(self, params: RepairParams) -> int:
        return int(max(params.lookback, params.min_streak) + 1)

    def indicators(self, data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        if params.lookback < 2 or params.min_streak < 1:
            raise ValueError("lookback must be >= 2 and min_streak >= 1")
        if not 0.0 <= params.min_repair < params.max_repair <= 1.0:
            raise ValueError("repair thresholds must satisfy 0 <= min < max <= 1")
        out = pd.DataFrame(index=data.index)
        out["descending"] = (
            data["high"].lt(data["high"].shift(1))
            & data["low"].lt(data["low"].shift(1))
        ).fillna(False)
        pressure_groups = (~out["descending"]).cumsum()
        out["pressure_streak"] = out["descending"].astype(int).groupby(pressure_groups).cumsum()
        out["rising_low"] = data["low"].gt(data["low"].shift(1)).fillna(False)
        recovery_groups = (~out["rising_low"]).cumsum()
        out["recovery_streak"] = out["rising_low"].astype(int).groupby(recovery_groups).cumsum()
        out["prior_ceiling"] = data["high"].shift(1).rolling(
            params.lookback, min_periods=params.lookback
        ).max()
        out["prior_floor"] = data["low"].shift(1)
        out["prior_pressure"] = out["pressure_streak"].shift(1)
        out["seed"] = (
            out["recovery_streak"].eq(1)
            & out["prior_pressure"].ge(params.min_streak)
            & out["prior_ceiling"].notna()
        )
        anchors = pd.DataFrame({
            "ceiling": out["prior_ceiling"].where(out["seed"]),
            "floor": out["prior_floor"].where(out["seed"]),
            "episode_pressure": out["prior_pressure"].where(out["seed"]),
        }, index=data.index).groupby(recovery_groups).ffill()
        out = out.join(anchors)
        out["depth"] = out["ceiling"] - out["floor"]
        out["repair"] = ((data["close"] - out["floor"]) / out["depth"].where(
            out["depth"].gt(0)
        )).replace([np.inf, -np.inf], np.nan)
        out["strength"] = (
            (out["episode_pressure"] / params.min_streak).clip(0.0, 2.0) / 2.0
            * (1.0 - out["repair"]).clip(0.0, 1.0)
        ).fillna(0.0)
        out["entry_condition"] = (
            out["rising_low"]
            & out["episode_pressure"].ge(params.min_streak)
            & out["repair"].ge(params.min_repair)
            & out["repair"].lt(params.max_repair)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RepairParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The position exits exactly when its entry condition becomes false.
        df["signal"] = indicators["entry_condition"].fillna(False).astype(int)
        df["size"] = (0.25 + 0.70 * indicators["strength"].fillna(0.0)).clip(0.25, 0.95)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    close = [105., 103., 101., 99., 95., 98., 100., 99., 95.]
    data = pd.DataFrame({
        "open": close,
        "high": [110., 108., 106., 104., 103., 104., 105., 106., 107.],
        "low": [100., 98., 96., 94., 92., 93., 94., 93., 92.],
        "close": close,
        "volume": 1000.,
    }, index=index)
    params = RepairParams(lookback=3, min_streak=2)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert result["size"].between(0.25, 0.95).all()
    assert result["size"].iloc[6] > result["size"].iloc[7]
    prefix = data.iloc[:7]
    prefix_result = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params
    ).data
    pd.testing.assert_frame_equal(prefix_result, result.iloc[:7])
    assert strategy.warmup_bars(params) >= params.lookback + 1


if __name__ == "__main__":
    _self_check()
