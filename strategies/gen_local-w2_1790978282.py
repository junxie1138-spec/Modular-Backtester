from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringParams:
    compression_ratio: float = 0.65
    cancellation_limit: float = 0.35
    energy_threshold: float = 0.20


class GeneratedStrategy(BaseStrategy[SpringParams]):
    strategy_id = "gen_local-w2_1790978282"

    @classmethod
    def params_type(cls):
        return SpringParams

    def warmup_bars(self, params: SpringParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: SpringParams) -> pd.DataFrame:
        if not (
            0.0 < params.compression_ratio <= 1.0
            and 0.0 < params.cancellation_limit < 1.0
            and 0.0 < params.energy_threshold < 1.0
        ):
            raise ValueError("Compression must be in (0, 1]; other thresholds in (0, 1).")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        out["squared_return"] = r.pow(2)
        out["signed_squared_return"] = r * r.abs()
        out["gross_return"] = r.abs().rolling(8, min_periods=8).sum()
        out["net_return"] = r.rolling(8, min_periods=8).sum()
        out["energy"] = out["squared_return"].rolling(8, min_periods=8).sum()
        out["signed_energy"] = out["signed_squared_return"].rolling(8, min_periods=8).sum()
        out["recent_abs_return"] = r.abs().rolling(3, min_periods=3).mean()
        out["prior_abs_return"] = r.abs().shift(3).rolling(5, min_periods=5).mean()
        out["recent_return"] = r.rolling(3, min_periods=3).sum()
        out["cancellation"] = out["net_return"].abs() / out["gross_return"].replace(0.0, np.nan)
        out["energy_skew"] = out["signed_energy"] / out["energy"].replace(0.0, np.nan)
        out["compressed"] = (
            (out["prior_abs_return"] > 0.0)
            & (out["recent_abs_return"] <= params.compression_ratio * out["prior_abs_return"])
        )
        eligible = (
            out["compressed"]
            & (out["cancellation"] <= params.cancellation_limit)
            & (out["energy_skew"].abs() >= params.energy_threshold)
            & (out["energy_skew"] * out["recent_return"] < 0.0)
            & (r * out["recent_return"] > 0.0)
        )
        out["entry_direction"] = np.select(
            [eligible & (out["energy_skew"] < 0.0), eligible & (out["energy_skew"] > 0.0)],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringParams,
    ) -> SignalFrame:
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SpringParams()
    index = pd.date_range("2020-01-01", periods=9)
    returns = np.array([0.0, -0.03, 0.01, 0.01, 0.008, 0.006, 0.002, 0.002, 0.002])
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({column: close for column in ("open", "high", "low", "close")}, index=index)
    data["volume"] = 1000.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert derived["entry_direction"].iloc[:8].eq(0).all()
    assert derived["entry_direction"].iloc[8] == 1
    mirrored = data.copy()
    mirrored_close = 100.0 * np.cumprod(1.0 - returns)
    for column in ("open", "high", "low", "close"):
        mirrored[column] = mirrored_close
    assert strategy.indicators(mirrored, params)["entry_direction"].iloc[8] == -1
    entries = pd.DataFrame({"entry_direction": [1, 0, 1, -1, 0, -1, 1, 1, 0]}, index=index)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, 0, 1]
    assert result["size"].gt(0.0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    assert strategy.indicators(flat, params)["entry_direction"].eq(0).all()
