from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    lookback: int = 40
    roc_bars: int = 2
    acceleration_z: float = 0.75
    rho_max: float = -0.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791290780"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    @staticmethod
    def warmup_bars(params: AccelerationParams) -> int:
        if params.lookback < 3 or params.roc_bars < 1:
            raise ValueError("Lookbacks must be positive; correlation requires at least three bars.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not np.isfinite(params.acceleration_z) or params.acceleration_z <= 0:
            raise ValueError("acceleration_z must be finite and positive.")
        if not np.isfinite(params.rho_max) or not -1 <= params.rho_max < 0:
            raise ValueError("rho_max must be in [-1, 0).")
        return int(params.lookback + params.roc_bars + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        out["roc"] = close.pct_change(periods=params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["lagged_acceleration"] = out["acceleration"].shift(1)
        out["acceleration_scale"] = out["lagged_acceleration"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0).replace(0.0, np.nan)
        out["acceleration_score"] = (out["acceleration"] / out["acceleration_scale"]).replace([np.inf, -np.inf], np.nan)
        out["acceleration_rho"] = out["acceleration"].rolling(params.lookback, min_periods=params.lookback).corr(out["lagged_acceleration"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["decay_regime"] = out["acceleration_rho"].le(params.rho_max)
        out["up_burst"] = out["decay_regime"] & out["roc"].gt(0) & out["acceleration_score"].ge(params.acceleration_z)
        out["down_burst"] = out["decay_regime"] & out["roc"].lt(0) & out["acceleration_score"].le(-params.acceleration_z)
        out["confirmed_up"] = out["up_burst"] & out["up_burst"].shift(1, fill_value=False)
        out["confirmed_down"] = out["down_burst"] & out["down_burst"].shift(1, fill_value=False)
        out["entry_direction"] = np.select([out["confirmed_up"], out["confirmed_down"]], [-1, 1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        target = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if position:
                if i >= exit_bar:
                    position = 0
                target[i] = position
                continue
            if candidate:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            target[i] = position
        df = pd.DataFrame({"signal": target, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_direction": [1, 1, -1, -1, -1, 0, 1, -1, 0, 0]}, index=index)
    for hold, expected in ((1, [0, 1, 0, -1, 0, -1, 0, 1, 0, 0]), (2, [0, 1, 1, 0, -1, -1, 0, 1, 1, 0])):
        result = GeneratedStrategy.generate_signals(data, indicators, None, AccelerationParams(hold_bars=hold))
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(data, AccelerationParams())
    assert flat["entry_direction"].eq(0).all()


if __name__ == "__main__":
    _self_check()
