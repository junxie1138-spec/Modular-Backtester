from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendTurnParams:
    fast_window: int = 5
    slow_window: int = 24
    channel_window: int = 12
    volatility_window: int = 20
    innovation_threshold: float = 0.15
    profit_target: float = 0.012
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[TrendTurnParams]):
    strategy_id = "gen_local_1790994070"

    @classmethod
    def params_type(cls) -> type[TrendTurnParams]:
        return TrendTurnParams

    def warmup_bars(self, params: TrendTurnParams) -> int:
        return int(max(
            max(2, params.fast_window),
            max(2, params.slow_window) + 1,
            max(2, params.channel_window) + 2,
            max(2, params.volatility_window) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: TrendTurnParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        for name, window in (
            ("fast_slope", max(2, params.fast_window)),
            ("slow_slope", max(2, params.slow_window)),
        ):
            weights = np.arange(window, dtype=float) - (window - 1) / 2.0
            denominator = float(np.square(weights).sum())
            values = out["log_close"].fillna(0.0).to_numpy()
            fitted = np.convolve(values, weights[::-1], mode="full")[:len(values)] if len(values) else values
            slope = pd.Series(fitted / denominator, index=data.index)
            out[name] = slope.where(out["log_close"].rolling(window).count() == window)
        out["prior_slow_slope"] = out["slow_slope"].shift(1)
        out["return_volatility"] = out["log_return"].rolling(
            max(2, params.volatility_window), min_periods=max(2, params.volatility_window)
        ).std(ddof=0)
        out["slope_innovation"] = out["fast_slope"] - out["prior_slow_slope"]
        out["innovation_score"] = (
            out["slope_innovation"] / out["return_volatility"].where(out["return_volatility"] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        channel = max(2, params.channel_window)
        out["channel_low"] = data["low"].rolling(channel, min_periods=channel).min().shift(1)
        out["channel_high"] = data["high"].rolling(channel, min_periods=channel).max().shift(1)
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (
            (close - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["prior_relative_position"] = out["relative_position"].shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendTurnParams,
    ) -> SignalFrame:
        threshold = max(float(params.innovation_threshold), 1e-6)
        position = indicators["relative_position"]
        previous = indicators["prior_relative_position"]
        score = indicators["innovation_score"]
        prior_trend = indicators["prior_slow_slope"]
        ready = indicators[[
            "relative_position", "prior_relative_position",
            "innovation_score", "prior_slow_slope",
        ]].notna().all(axis=1)
        long_entry = ready & (previous <= 0.5) & (position > 0.5) & (score >= threshold) & (prior_trend < 0)
        short_entry = ready & (previous >= 0.5) & (position < 0.5) & (score <= -threshold) & (prior_trend > 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        minimum = float(np.clip(params.min_size, 0.01, 1.0))
        maximum = float(np.clip(params.max_size, minimum, 1.0))
        conviction = ((score.abs() - threshold) / (2.0 * threshold)).clip(0.0, 1.0).fillna(0.0)
        entry_sizes = (minimum + (maximum - minimum) * conviction).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum, dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        trade_size = minimum
        holding_limit = max(1, int(params.hold_bars))
        target = max(float(params.profit_target), 1e-6)
        for i in range(len(data)):
            if side != 0:
                # A decision at the previous close enters at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                expired = i - entry_bar + 1 >= holding_limit
                if not valid_price or not np.isfinite(closes[i]) or gain >= target or expired:
                    side = 0
                    entry_price = np.nan
                    trade_size = minimum
            elif candidates[i] != 0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                trade_size = float(entry_sizes[i])
            signals[i] = side
            sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(minimum).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
