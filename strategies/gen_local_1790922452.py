from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    lookback: int = 8
    acceleration_lag: int = 3
    acceleration_threshold: float = 0.002
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.lookback, self.acceleration_lag, self.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not np.isfinite(self.acceleration_threshold) or self.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local_1790922452"

    @classmethod
    def params_type(cls):
        return GapAccelerationParams

    def warmup_bars(self, params: GapAccelerationParams) -> int:
        return int(params.lookback + params.acceleration_lag + 2)

    def indicators(self, data: pd.DataFrame, params: GapAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap_log_return"] = np.log(opening / closing.shift(1))
        out["intraday_log_return"] = np.log(closing / opening)
        out["overnight_roc"] = np.expm1(
            out["gap_log_return"].rolling(params.lookback, min_periods=params.lookback).sum()
        )
        out["intraday_roc"] = np.expm1(
            out["intraday_log_return"].rolling(params.lookback, min_periods=params.lookback).sum()
        )
        out["gap_acceleration"] = out["overnight_roc"].diff(params.acceleration_lag)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAccelerationParams,
    ) -> SignalFrame:
        valid = indicators[["overnight_roc", "intraday_roc", "gap_acceleration"]].notna().all(axis=1)
        long_entry = (
            valid
            & (indicators["overnight_roc"] > 0)
            & (indicators["intraday_roc"] < 0)
            & (indicators["gap_acceleration"] > params.acceleration_threshold)
        )
        short_entry = (
            valid
            & (indicators["overnight_roc"] < 0)
            & (indicators["intraday_roc"] > 0)
            & (indicators["gap_acceleration"] < -params.acceleration_threshold)
        )
        candidate = pd.Series(
            np.select([long_entry, short_entry], [1, -1], default=0),
            index=data.index,
            dtype=int,
        )
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(confirmed.to_numpy()):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapAccelerationParams()
    index = pd.date_range("2020-01-01", periods=params.holding_bars + 5, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    assert strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].eq(0).all()
    for direction in (-1, 1):
        direction_by_bar = np.full(len(data), -direction, dtype=int)
        direction_by_bar[:2] = direction
        indicators = pd.DataFrame(
            {
                "overnight_roc": direction_by_bar * 0.01,
                "intraday_roc": direction_by_bar * -0.01,
                "gap_acceleration": direction_by_bar * 0.01,
            },
            index=index,
        )
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[:2].eq(0).all()
        assert result["signal"].iloc[2:2 + params.holding_bars].eq(direction).all()
        assert result["signal"].iloc[2 + params.holding_bars] == 0
        assert result["signal"].iloc[3 + params.holding_bars] == -direction
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
