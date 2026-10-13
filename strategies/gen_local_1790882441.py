from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    fractional_order: float = 0.5
    compression_ratio: float = 0.75


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1790882441"

    @classmethod
    def params_type(cls):
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        return 54

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        weights = np.ones(32, dtype=float)
        for k in range(1, len(weights)):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed 32-term fractional filter; extend if tail sensitivity matters.
        out["fractional_close"] = sum(weights[k] * out["log_close"].shift(k) for k in range(len(weights)))
        out["fractional_impulse"] = out["fractional_close"].diff()
        out["impulse_scale"] = out["fractional_impulse"].rolling(20, min_periods=20).std(ddof=0).shift(1)
        out["impulse_score"] = out["fractional_impulse"] / out["impulse_scale"].where(out["impulse_scale"] > 0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["relative_range"] = out["true_range"] / previous_close.where(previous_close > 0)
        out["recent_range"] = out["relative_range"].rolling(3, min_periods=3).mean()
        out["baseline_range"] = out["relative_range"].rolling(20, min_periods=20).median().shift(3)
        out["compression"] = out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        valid = np.isfinite(out[["impulse_score", "compression", "log_return"]]).all(axis=1)
        qualified = valid & (out["compression"] < params.compression_ratio) & (out["impulse_score"].abs() > 0.5) & (out["fractional_impulse"] * out["log_return"] < 0)
        out["entry_direction"] = np.select([
            qualified & (out["fractional_impulse"] > 0),
            qualified & (out["fractional_impulse"] < 0),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoilParams) -> SignalFrame:
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position and i == exit_bar:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
                exit_bar = i + 2
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RecoilParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_direction": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry_direction"].eq(0).all()
