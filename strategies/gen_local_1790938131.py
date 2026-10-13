from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fd_window: int = 12
    fractional_order: float = 0.45
    vol_window: int = 10
    entry_band: float = 0.15
    wick_band: float = 0.05
    target_daily_vol: float = 0.006
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790938131"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.fd_window, params.vol_window) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.fd_window < 2 or params.vol_window < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one.")
        if not np.isfinite(params.entry_band) or params.entry_band < 0.0:
            raise ValueError("entry_band must be finite and nonnegative.")
        if not 0.0 <= params.wick_band < 1.0:
            raise ValueError("wick_band must be in [0, 1).")
        if not np.isfinite(params.target_daily_vol) or params.target_daily_vol <= 0.0:
            raise ValueError("target_daily_vol must be finite and positive.")
        if not 0.0 < params.max_size <= 1.0:
            raise ValueError("max_size must be in (0, 1].")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0.0)
        out["return"] = close.pct_change(fill_method=None)
        out["daily_vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0)

        weights = np.ones(params.fd_window, dtype=float)
        for k in range(1, params.fd_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed memory truncation; increase fd_window for longer memory.
        fractional_close = close.copy()
        for k in range(1, params.fd_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out["fractional_close"] = fractional_close
        out["fractional_move"] = fractional_close.diff() / close.shift(1)
        out["impulse"] = out["fractional_move"] / out["daily_vol"].where(
            out["daily_vol"] > 1e-12
        )

        out["bar_range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["lower_wick"] = (
            data[["open", "close"]].min(axis=1, skipna=False) - data["low"]
        ).clip(lower=0.0)
        out["upper_wick"] = (
            data["high"] - data[["open", "close"]].max(axis=1, skipna=False)
        ).clip(lower=0.0)
        out["wick_balance"] = (
            out["lower_wick"] - out["upper_wick"]
        ) / out["bar_range"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        impulse = indicators["impulse"]
        wick = indicators["wick_balance"]
        ready = np.isfinite(indicators[["impulse", "wick_balance", "daily_vol"]]).all(axis=1)
        long_entry = ready & (impulse > params.entry_band) & (wick > params.wick_band)
        short_entry = ready & (impulse < -params.entry_band) & (wick < -params.wick_band)
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))

        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposite two-primitive agreement exits to flat before re-entry.
                position = 0
            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = params.target_daily_vol / indicators["daily_vol"].where(
            indicators["daily_vol"] > 1e-12
        )
        exposure = exposure.replace([np.inf, -np.inf], np.nan).clip(
            lower=min(0.05, params.max_size), upper=params.max_size
        )
        df["size"] = exposure.shift(1).fillna(min(0.05, params.max_size)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "impulse": [1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, np.nan],
        "wick_balance": [1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, np.nan],
        "daily_vol": 0.01,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= max(params.fd_window, params.vol_window) + 1


if __name__ == "__main__":
    _self_check()
