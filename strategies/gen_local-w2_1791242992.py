from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetryParams:
    range_lookback: int = 4
    shock_multiple: float = 2.0
    quiet_fraction: float = 0.5
    retry_min: float = 0.35
    holding_bars: int = 7


class GeneratedStrategy(BaseStrategy[GapRetryParams]):
    strategy_id = "gen_local-w2_1791242992"

    @classmethod
    def params_type(cls) -> type[GapRetryParams]:
        return GapRetryParams

    def warmup_bars(self, params: GapRetryParams) -> int:
        if not 1 <= params.range_lookback <= 5:
            raise ValueError("range_lookback must be between 1 and 5")
        if not 5 <= params.holding_bars <= 10:
            raise ValueError("holding_bars must be between 5 and 10")
        if not 1.0 < params.shock_multiple < np.inf:
            raise ValueError("shock_multiple must be finite and greater than 1")
        if not 0.0 < params.quiet_fraction < 1.0:
            raise ValueError("quiet_fraction must be between 0 and 1")
        if not 0.0 < params.retry_min < 1.0:
            raise ValueError("retry_min must be between 0 and 1")
        return int(params.range_lookback + 5)

    def indicators(self, data: pd.DataFrame, params: GapRetryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        gap = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        range_fraction = ((data["high"] - data["low"]) / previous_close).replace([np.inf, -np.inf], np.nan)
        scale = range_fraction.rolling(params.range_lookback, min_periods=params.range_lookback).mean().shift(1)
        valid = gap.notna() & scale.notna() & scale.gt(0.0)
        shock_direction = pd.Series(
            np.where(valid & gap.abs().ge(params.shock_multiple * scale), np.sign(gap), 0),
            index=data.index,
            dtype=int,
        )
        quiet = valid & gap.abs().le(params.quiet_fraction * scale)
        anchor_direction = shock_direction.shift(3).fillna(0).astype(int)
        quiet_interval = quiet.shift(1, fill_value=False) & quiet.shift(2, fill_value=False)
        rejected = ((gap > 0.0) & data["close"].lt(previous_close)) | ((gap < 0.0) & data["close"].gt(previous_close))
        eligible = (
            valid
            & anchor_direction.ne(0)
            & quiet_interval
            & (gap * anchor_direction).gt(0.0)
            & gap.abs().ge(params.retry_min * scale)
            & gap.abs().le(scale)
            & rejected
        )
        out["previous_close"] = previous_close
        out["gap"] = gap
        out["range_fraction"] = range_fraction
        out["scale"] = scale
        out["valid"] = valid
        out["shock_direction"] = shock_direction
        out["quiet"] = quiet
        out["anchor_direction"] = anchor_direction
        out["quiet_interval"] = quiet_interval
        out["rejected"] = rejected
        out["eligible"] = eligible
        out["candidate"] = np.where(eligible, -anchor_direction, 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRetryParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_at:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                exit_at = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapRetryParams(holding_bars=5)
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index) and not computed.isna().any().any()
    candidates = pd.DataFrame({"candidate": [1, 0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, candidates, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0, 0]
    assert result["size"].gt(0.0).all()
    assert strategy.warmup_bars(GapRetryParams(range_lookback=5)) == 10
