from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRetentionParams:
    lookback: int = 6
    atr_period: int = 14
    entry_threshold: float = 0.04
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeRetentionParams]):
    strategy_id = "gen_local-w2_1791259987"

    @classmethod
    def params_type(cls):
        return RangeRetentionParams

    def warmup_bars(self, params: RangeRetentionParams) -> int:
        return int(max(params.lookback, params.atr_period) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeRetentionParams) -> pd.DataFrame:
        if not (
            params.lookback >= 1 and params.atr_period >= 1
            and params.hold_bars >= 1 and params.entry_threshold > 0
            and params.stop_atr > 0 and 0 < params.min_size <= params.max_size <= 1
        ):
            raise ValueError("Invalid range-retention parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["upper_addition"] = (data["high"] - out["previous_high"]).clip(lower=0)
        out["lower_addition"] = (out["previous_low"] - data["low"]).clip(lower=0)
        out["retained_upper"] = np.minimum(
            (data["close"] - out["previous_high"]).clip(lower=0),
            out["upper_addition"],
        )
        out["upper_sum"] = out["upper_addition"].rolling(params.lookback).sum()
        out["lower_sum"] = out["lower_addition"].rolling(params.lookback).sum()
        out["retained_sum"] = out["retained_upper"].rolling(params.lookback).sum()
        out["range_sum"] = out["range"].rolling(params.lookback).sum()
        out["retention"] = out["retained_sum"] / out["upper_sum"].replace(0, np.nan)
        out["discovery_balance"] = (
            (out["upper_sum"] - out["lower_sum"])
            / out["range_sum"].replace(0, np.nan)
        )
        out["score"] = (out["retention"] * out["discovery_balance"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRetentionParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        fixed_stop = np.nan
        entry_size = params.min_size
        for i in range(n):
            if active:
                if i == entry_bar:
                    # Anchor only when the next open becomes observable.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                if (
                    not np.isfinite(closes[i]) or not np.isfinite(fixed_stop)
                    or closes[i] < fixed_stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
                    continue
            elif (
                np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(scores[i]) and scores[i] >= params.entry_threshold
            ):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                conviction = np.clip(scores[i] / (3.0 * params.entry_threshold), 0.0, 1.0)
                entry_size = params.min_size + (params.max_size - params.min_size) * conviction
            if active:
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeRetentionParams()
    index = pd.date_range("2020-01-01", periods=7)
    close = np.array([100, 100, 101, 105, 101, 97, 97], dtype=float)
    data = pd.DataFrame({
        "open": 100.0, "high": np.maximum(close, 100) + 1,
        "low": np.minimum(close, 100) - 1, "close": close, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    ind = pd.DataFrame({
        "score": [0, 0.08, 0, 0, 0, 0, 0],
        "atr": [1, 1, 50, 50, 50, 50, 50],
    }, index=index)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    stronger = ind.copy()
    stronger.loc[index[1], "score"] = 0.12
    assert strategy.generate_signals(data, stronger, ctx, params).data["size"].iloc[2] > result["size"].iloc[2]
    prefix = strategy.generate_signals(data.iloc[:5], ind.iloc[:5], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:5])
    assert strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].eq(0).all()
    expiry = RangeRetentionParams(hold_bars=2)
    assert strategy.generate_signals(data, ind, ctx, expiry).data["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
