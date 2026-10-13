from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.4
    compression_ratio: float = 0.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791245633"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 40

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 < params.compression_ratio <= 1.0:
            raise ValueError("compression_ratio must be between zero and one inclusive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        weights = np.ones(16)
        for k in range(1, 16):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed 16-lag fractional approximation; extend if longer memory is needed.
        out["fractional_close"] = sum(
            weights[k] * out["log_close"].shift(k) for k in range(16)
        )
        out["fd_center"] = out["fractional_close"].rolling(4, min_periods=4).mean()
        out["fd_dislocation"] = out["fractional_close"] - out["fd_center"]
        out["fd_scale"] = out["fd_dislocation"].rolling(20, min_periods=20).std(ddof=0)
        out["zscore"] = out["fd_dislocation"] / out["fd_scale"].where(out["fd_scale"] > 1e-12)
        out["normalized_range"] = (
            (data["high"] - data["low"]).where(data["high"] >= data["low"])
            / data["close"].where(data["close"] > 0)
        )
        out["fast_range"] = out["normalized_range"].rolling(4, min_periods=4).mean()
        out["slow_range"] = out["normalized_range"].rolling(20, min_periods=20).mean()
        out["compression"] = out["fast_range"] / out["slow_range"].where(out["slow_range"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        z = indicators["zscore"]
        compressed = indicators["compression"].le(params.compression_ratio)
        valid = np.isfinite(z) & np.isfinite(indicators["compression"])
        candidate = np.select(
            [valid & compressed & z.le(-0.75), valid & compressed & z.ge(0.75)],
            [1, -1],
            default=0,
        )
        raw = np.zeros(len(data), dtype=int)
        state = 0
        for i, side in enumerate(candidate):
            if state == 0:
                state = int(side)
            elif side == -state:
                state = 0
            raw[i] = state
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"zscore": [-1.0, -1.0, 1.0, 1.0, 0.0, -1.0, np.nan], "compression": 0.5},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=40)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat_index = pd.date_range("2020-01-01", periods=80)
    flat = data.iloc[:1].reindex(flat_index).ffill()
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_indicators, ctx, params).data["signal"].eq(0).all()
