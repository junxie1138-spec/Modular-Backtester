from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryReleaseParams:
    lookback: int = 5
    min_down_days: int = 3
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExpiryReleaseParams]):
    strategy_id = "gen_local_1790871949"

    @classmethod
    def params_type(cls):
        return ExpiryReleaseParams

    def warmup_bars(self, params: ExpiryReleaseParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not 1 <= params.min_down_days <= params.lookback:
            raise ValueError("min_down_days must be between 1 and lookback")
        if not 15 <= params.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: ExpiryReleaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        net_return = returns.rolling(params.lookback, min_periods=params.lookback).sum()
        down_days = (returns < 0).astype(float).where(returns.notna()).rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        expiry_monday = pd.Series(
            (data.index.dayofweek == 0) & (data.index.day >= 18) & (data.index.day <= 24),
            index=data.index,
        )
        out = pd.DataFrame(index=data.index)
        out["return"] = returns.fillna(0.0)
        out["net_return"] = net_return.fillna(0.0)
        out["down_days"] = down_days.fillna(0.0)
        out["expiry_monday"] = expiry_monday
        out["ready"] = net_return.notna() & down_days.notna()
        out["entry"] = (
            out["ready"] & expiry_monday & (out["return"] > 0)
            & (out["net_return"] < 0) & (out["down_days"] >= params.min_down_days)
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ExpiryReleaseParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ExpiryReleaseParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2024-01-01", periods=64)
    prices = 100.0 + np.arange(len(index), dtype=float)
    data = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1,
         "close": prices, "volume": 1000.0}, index=index,
    )
    indicators = strategy.indicators(data, params)
    assert not indicators["entry"].any()
    assert strategy.warmup_bars(ExpiryReleaseParams(lookback=9)) == 10
    indicators["entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    expected = (np.arange(len(data)) > 0) & (
        (np.maximum(np.arange(len(data)) - 1, 0) % (params.hold_bars + 1)) < params.hold_bars
    )
    assert np.array_equal(signals["signal"].to_numpy(), expected.astype(int))
    assert signals["size"].gt(0).all()
