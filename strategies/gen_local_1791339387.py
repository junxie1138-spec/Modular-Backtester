from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_multiple: float = 1.15
    min_efficiency: float = 0.55
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791339387"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 6

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive")
        if not np.isfinite(params.min_efficiency) or not 0 < params.min_efficiency <= 1:
            raise ValueError("min_efficiency must be in (0, 1]")

        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        prior_volume = data["volume"].shift(1).rolling(5, min_periods=5).mean()
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        month = pd.Series(data.index.year * 12 + data.index.month, index=data.index)

        out["gap_return"] = data["open"].div(previous_close).sub(1)
        out["session_return"] = data["close"].div(opening).sub(1)
        out["efficiency"] = (data["close"] - data["open"]).abs().div(span)
        out["volume_ratio"] = data["volume"].div(prior_volume.where(prior_volume > 0))
        out["month_start"] = month.ne(month.shift(1)) & month.shift(1).notna()
        ready = out[["gap_return", "session_return", "efficiency", "volume_ratio"]].notna().all(axis=1)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
        agreement = out["gap_return"].mul(out["session_return"]).gt(0)
        eligible = (
            ready
            & out["month_start"].astype(bool)
            & agreement
            & out["efficiency"].ge(params.min_efficiency)
            & out["volume_ratio"].ge(params.volume_multiple)
        )
        out["entry"] = np.where(eligible, np.sign(out["session_return"]), 0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not isinstance(params.hold_bars, int) or isinstance(params.hold_bars, bool) or not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be an integer from 15 through 20")

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i, entry in enumerate(entries):
            if position != 0:
                if i >= exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=45)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000.0},
        index=index,
    )
    params = Params()
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index) and not derived.isna().any().any()
    assert GeneratedStrategy.warmup_bars(params) <= 10
    for direction in (-1, 1):
        entries = pd.DataFrame({"entry": direction}, index=index)
        result = GeneratedStrategy.generate_signals(data, entries, None, params).data
        assert result["signal"].iloc[0] == 0
        assert result["signal"].iloc[1:19].eq(direction).all()
        assert result["signal"].iloc[19] == 0
        assert result["signal"].iloc[20] == direction
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
