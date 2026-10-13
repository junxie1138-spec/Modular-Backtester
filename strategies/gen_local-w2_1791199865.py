from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    window: int = 4
    migration_threshold: float = 0.35
    pressure_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = "gen_local-w2_1791199865"

    @classmethod
    def params_type(cls) -> type[MigrationParams]:
        return MigrationParams

    def warmup_bars(self, params: MigrationParams) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        if params.window < 1:
            raise ValueError("window must be positive")
        if not 0.0 <= params.migration_threshold <= 1.0:
            raise ValueError("migration_threshold must be between zero and one")
        if not 0.0 <= params.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["high_step"] = data["high"].diff()
        out["low_step"] = data["low"].diff()
        out["boundary_move"] = out["high_step"] + out["low_step"]
        out["boundary_travel"] = out["high_step"].abs() + out["low_step"].abs()
        out["net_migration"] = out["boundary_move"].rolling(params.window, min_periods=params.window).sum()
        out["total_travel"] = out["boundary_travel"].rolling(params.window, min_periods=params.window).sum()
        out["migration_efficiency"] = out["net_migration"] / out["total_travel"].where(out["total_travel"] > 0.0)
        out["daily_range"] = data["high"] - data["low"]
        out["candle_pressure"] = (data["close"] - data["open"]) / out["daily_range"].where(out["daily_range"] > 0.0)
        out["weighted_pressure"] = out["candle_pressure"] * data["volume"]
        out["pressure_sum"] = out["weighted_pressure"].rolling(params.window, min_periods=params.window).sum()
        out["volume_sum"] = data["volume"].rolling(params.window, min_periods=params.window).sum()
        out["volume_pressure"] = out["pressure_sum"] / out["volume_sum"].where(out["volume_sum"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        agreement = (
            (indicators["migration_efficiency"] > params.migration_threshold)
            & (indicators["volume_pressure"] > params.pressure_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Losing either entry primitive reverses eligibility and exits.
        df["signal"] = agreement.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    close = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9], dtype=float)
    data = pd.DataFrame({
        "open": close - np.where(np.arange(12) < 6, 0.75, -0.75),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 100.0,
    }, index=pd.date_range("2020-01-01", periods=12))
    params = MigrationParams(window=2)
    indicators = GeneratedStrategy.indicators(None, data, params)
    signals = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert signals.index.equals(data.index)
    assert (signals["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
