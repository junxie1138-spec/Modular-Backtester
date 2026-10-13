from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 15
    selloff_streak: int = 3
    min_drawdown: float = 0.025
    repair_fraction: float = 0.5


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791287547"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.selloff_streak + 1) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.selloff_streak < 2:
            raise ValueError("Lookback and selloff streak must be at least two.")
        if not 0 < params.min_drawdown < 1 or not 0 < params.repair_fraction < 1:
            raise ValueError("Drawdown and repair fractions must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["lower_low"] = data["low"].lt(data["low"].shift(1))
        resets = (~out["lower_low"]).cumsum()
        out["selloff_count"] = out["lower_low"].astype(int).groupby(resets).cumsum()
        out["recent_peak"] = data["high"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - data["low"].div(out["recent_peak"].where(out["recent_peak"].gt(0)))
        out["anchor_event"] = out["selloff_count"].ge(params.selloff_streak) & out["drawdown"].ge(params.min_drawdown)
        out["anchor_peak"] = out["recent_peak"].where(out["anchor_event"]).ffill()
        out["anchor_low"] = data["low"].where(out["anchor_event"]).ffill()
        out["repair_level"] = out["anchor_low"] + params.repair_fraction * (out["anchor_peak"] - out["anchor_low"])
        out["recovery_condition"] = (
            data["close"].gt(out["repair_level"])
            & data["close"].lt(out["anchor_peak"])
            & out["repair_level"].notna()
            & data[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
        )
        resets = (~out["recovery_condition"]).cumsum()
        out["recovery_count"] = out["recovery_condition"].astype(int).groupby(resets).cumsum()
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Two corridor closes confirm entry; the first corridor failure exits.
        df["signal"] = indicators["recovery_count"].ge(2).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    close = pd.Series([100, 99, 98, 97, 96, 97, 98, 99, 97, 96, 96.3], index=pd.date_range("2020-01-01", periods=11), dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0})
    params = RecoveryParams(peak_window=4, min_drawdown=0.02, repair_fraction=0.35)
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    prefix = data.iloc[:8]
    assert GeneratedStrategy.indicators(prefix, params).equals(derived.iloc[:8])
    assert result.index.equals(data.index) and result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= max(params.peak_window, params.selloff_streak + 1)


if __name__ == "__main__":
    _self_check()
