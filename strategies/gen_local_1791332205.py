from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.55
    shift_threshold: float = 1.8
    min_efficiency: float = 0.7
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791332205"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0 < params.fractional_order < 1:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.shift_threshold) or params.shift_threshold <= 0:
            raise ValueError("shift_threshold must be finite and positive")
        if not 0 <= params.min_efficiency <= 1:
            raise ValueError("min_efficiency must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        weight = 1.0
        out["fractional_close"] = close
        # ponytail: four-tap truncation; extend only if the ten-bar warmup limit changes.
        for lag in range(1, 4):
            weight *= (lag - 1 - params.fractional_order) / lag
            out["fractional_close"] += weight * close.shift(lag)
        fd = out["fractional_close"]
        out["recent_mean"] = fd.rolling(2, min_periods=2).mean()
        out["baseline_mean"] = fd.shift(2).rolling(4, min_periods=4).mean()
        out["baseline_std"] = fd.shift(2).rolling(4, min_periods=4).std(ddof=0)
        out["fractional_change"] = fd.diff()
        out["path_length"] = out["fractional_change"].abs().rolling(2, min_periods=2).sum()
        out["efficiency"] = ((fd - fd.shift(2)) / out["path_length"].replace(0, np.nan)).clip(-1, 1)
        out["shift_score"] = ((out["recent_mean"] - out["baseline_mean"]) / out["baseline_std"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["recent_floor"] = fd.rolling(2, min_periods=2).min()
        out["entry"] = (
            out["shift_score"].gt(params.shift_threshold)
            & out["shift_score"].shift(1).le(params.shift_threshold)
            & out["recent_floor"].gt(out["baseline_mean"])
            & out["efficiency"].ge(params.min_efficiency)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not isinstance(params.hold_bars, int) or isinstance(params.hold_bars, bool) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(hold_bars=7)
    index = pd.date_range("2020-01-01", periods=24)
    close = pd.Series(100.0 + np.arange(24) ** 1.2, index=index)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000})
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:16], params)
    pd.testing.assert_frame_equal(full.iloc[:16], prefix)
    assert not full["entry"].iloc[:9].any()
    entries = pd.DataFrame({"entry": True}, index=index)
    result = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 7 + [0] + [1] * 7 + [0] + [1] * 7
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
