from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorAccelerationParams:
    acceleration_threshold: float = 0.4
    floor_share: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FloorAccelerationParams]):
    strategy_id = "gen_local-w2_1791246533"

    @classmethod
    def params_type(cls) -> type[FloorAccelerationParams]:
        return FloorAccelerationParams

    def warmup_bars(self, params: FloorAccelerationParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: FloorAccelerationParams) -> pd.DataFrame:
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.floor_share) or params.floor_share <= 0:
            raise ValueError("floor_share must be finite and positive")
        close = data["close"].where(data["close"] > 0)
        low = data["low"].where(data["low"] > 0)
        out = pd.DataFrame(index=data.index)
        out["daily_return"] = close.pct_change(fill_method=None)
        out["noise"] = out["daily_return"].rolling(5, min_periods=5).std(ddof=0)
        out["close_roc"] = close.pct_change(3, fill_method=None)
        out["low_roc"] = low.pct_change(3, fill_method=None)
        out["close_acceleration"] = out["close_roc"].diff()
        out["low_acceleration"] = out["low_roc"].diff()
        out["scaled_acceleration"] = out["close_acceleration"] / out["noise"].where(out["noise"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["close_roc"] > 0)
            & (out["low_roc"] > 0)
            & (out["scaled_acceleration"] >= params.acceleration_threshold)
            & (out["low_acceleration"] >= params.floor_share * out["close_acceleration"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FloorAccelerationParams,
    ) -> SignalFrame:
        if isinstance(params.hold_bars, bool) or params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_index = -1
        for i in range(len(data)):
            if exit_index >= 0:
                if i < exit_index:
                    signal[i] = 1
                else:
                    # Keep the scheduled exit flat even when another entry qualifies.
                    exit_index = -1
            elif entries[i]:
                exit_index = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=24, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    strategy = GeneratedStrategy()
    for hold in (3, 4, 5):
        params = FloorAccelerationParams(hold_bars=hold)
        result = strategy.generate_signals(data, indicators, None, params).data
        expected = [int(i % (hold + 1) != 0) for i in range(len(data))]
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
        assert strategy.warmup_bars(params) == 6


if __name__ == "__main__":
    _self_check()
