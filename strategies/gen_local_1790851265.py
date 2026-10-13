from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReentryParams:
    channel_bars: int = 20
    percentile_bars: int = 126
    gap_percentile: float = 0.85
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapReentryParams]):
    strategy_id = "gen_local_1790851265"

    @classmethod
    def params_type(cls):
        return GapReentryParams

    def warmup_bars(self, params: GapReentryParams) -> int:
        return int(max(params.channel_bars + 1, params.atr_bars + params.percentile_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapReentryParams) -> pd.DataFrame:
        if min(params.channel_bars, params.percentile_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0.0 < params.gap_percentile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Invalid percentile or stop multiplier.")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["channel_high"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        out["gap"] = data["open"] - prior_close
        out["gap_strength"] = out["gap"].abs() / out["atr"].shift(1).where(out["atr"].shift(1) > 0.0)
        out["gap_threshold"] = out["gap_strength"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.gap_percentile).shift(1)
        out["open_position"] = (data["open"] - out["channel_low"]) / (
            out["channel_high"] - out["channel_low"]
        ).replace(0.0, np.nan)
        out["close_position"] = (data["close"] - out["channel_low"]) / (
            out["channel_high"] - out["channel_low"]
        ).replace(0.0, np.nan)
        valid = np.isfinite(out).all(axis=1) & (out["atr"] > 0.0)
        extreme = out["gap_strength"] > out["gap_threshold"]
        inside = out["close_position"].between(0.0, 1.0, inclusive="neither")
        long_setup = valid & extreme & inside & (out["open_position"] < 0.0) & (out["gap"] < 0.0) & (data["close"] > data["open"])
        short_setup = valid & extreme & inside & (out["open_position"] > 1.0) & (out["gap"] > 0.0) & (data["close"] < data["open"])
        out["setup"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReentryParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        age = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            if position:
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_atr = atr[i - 1]
                    age = 0
                age += 1
                stopped = position * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                invalid = not np.isfinite(entry_price) or not np.isfinite(entry_atr) or not np.isfinite(closes[i])
                if stopped or invalid or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    entry_atr = np.nan
                signals[i] = position
            elif setups[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(setups[i])
                entry_price = np.nan
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapReentryParams()
    index = pd.date_range("2020-01-01", periods=9)
    prices = pd.DataFrame({
        "open": [100, 110, 111, 109, 100, 100, 101, 103, 100],
        "close": [100, 110, 111, 107, 100, 101, 103, 100, 100],
    }, index=index)
    derived = pd.DataFrame({"atr": 1.0, "setup": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(prices, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 0]
    timed = strategy.generate_signals(prices, derived, ctx, GapReentryParams(stop_atr=100.0, hold_bars=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
