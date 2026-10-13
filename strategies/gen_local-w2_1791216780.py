from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    half_window: int = 5
    rank_window: int = 126
    history_window: int = 252
    hold_bars: int = 18
    atr_window: int = 14
    similarity: float = 0.15
    min_analogs: int = 12
    entry_percentile: float = 0.80
    trail_atr: float = 3.0

    def __post_init__(self):
        windows = (self.half_window, self.rank_window, self.history_window,
                   self.hold_bars, self.atr_window, self.min_analogs)
        if any(x < 1 for x in windows):
            raise ValueError("Windows and analog count must be positive.")
        if self.min_analogs > self.history_window:
            raise ValueError("min_analogs exceeds history_window.")
        if not (0 < self.similarity <= 1 and 0.5 < self.entry_percentile < 1):
            raise ValueError("Invalid percentile parameters.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791216780"

    @classmethod
    def params_type(cls):
        return AnalogParams

    @classmethod
    def warmup_bars(cls, params):
        return int(max(2 * params.half_window + 1, params.atr_window + 1)
                   + 2 * params.rank_window + params.hold_bars
                   + params.history_window)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window,
                                min_periods=params.atr_window).mean()
        out["half_return"] = close / close.shift(params.half_window) - 1.0
        out["curvature"] = out["half_return"] - out["half_return"].shift(params.half_window)
        volume = data["volume"].where(data["volume"] >= 0)
        recent_volume = volume.rolling(params.half_window).sum()
        total_volume = volume.rolling(2 * params.half_window).sum()
        out["late_volume_share"] = recent_volume / total_volume.where(total_volume > 0)
        for source in ("curvature", "late_volume_share"):
            out[source + "_rank"] = out[source].rolling(
                params.rank_window, min_periods=params.rank_window
            ).rank(pct=True)
        out["matured_return"] = close / close.shift(params.hold_bars) - 1.0
        if len(data) == 0:
            for name in ("analog_count", "forecast", "forecast_rank"):
                out[name] = pd.Series(index=data.index, dtype=float)
            return out
        # Each historical feature is paired only with an already completed outcome.
        def history(series):
            padded = np.pad(series.to_numpy(dtype=float),
                            (params.history_window - 1, 0),
                            constant_values=np.nan)
            return np.lib.stride_tricks.sliding_window_view(
                padded, params.history_window
            )
        a = out["curvature_rank"]
        b = out["late_volume_share_rank"]
        past_a = history(a.shift(params.hold_bars))
        past_b = history(b.shift(params.hold_bars))
        outcomes = history(out["matured_return"])
        matches = ((np.abs(past_a - a.to_numpy()[:, None]) <= params.similarity)
                   & (np.abs(past_b - b.to_numpy()[:, None]) <= params.similarity)
                   & np.isfinite(outcomes))
        counts = matches.sum(axis=1)
        totals = np.where(matches, outcomes, 0.0).sum(axis=1)
        forecast = np.full(len(data), np.nan)
        np.divide(totals, counts, out=forecast,
                  where=counts >= params.min_analogs)
        out["analog_count"] = counts
        out["forecast"] = forecast
        out["forecast_rank"] = out["forecast"].rolling(
            params.rank_window, min_periods=max(1, params.rank_window // 4)
        ).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        long_setup = ((indicators["forecast"] > 0)
                      & (indicators["forecast_rank"] >= params.entry_percentile))
        short_setup = ((indicators["forecast"] < 0)
                       & (indicators["forecast_rank"] <= 1 - params.entry_percentile))
        candidate = pd.Series(np.select([long_setup, short_setup], [1, -1], 0),
                              index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = np.nan
        stop = -np.inf
        short_ceiling = np.nan
        entry_distance = np.nan
        for i in range(len(data)):
            price = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
                position = 0
                continue
            if position:
                if i == entry_bar:
                    high_water = price
                    short_ceiling = float(data["open"].iloc[i]) + entry_distance
                high_water = max(high_water, price)
                # The close-based floor ratchets upward even when ATR expands.
                stop = max(stop, high_water - params.trail_atr * atr)
                expired = i - entry_bar + 1 >= params.hold_bars
                # Shorts also have a fixed entry-ATR risk ceiling.
                short_loss = position == -1 and price >= short_ceiling
                if price <= stop or expired or short_loss:
                    position = 0
            elif i >= self.warmup_bars(params) - 1 and confirmed.iloc[i] != 0:
                position = int(confirmed.iloc[i])
                entry_bar = i + 1
                high_water = np.nan
                stop = -np.inf
                entry_distance = params.trail_atr * atr
                short_ceiling = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams(half_window=1, rank_window=1, history_window=1,
                         hold_bars=18, atr_window=1, min_analogs=1, trail_atr=2.0)
    strategy = object.__new__(GeneratedStrategy)
    offset = strategy.warmup_bars(params)
    close = [100.0] * offset + [100.0, 100.0, 100.0, 110.0, 109.0, 106.0, 105.0]
    index = pd.date_range("2020-01-01", periods=len(close))
    data = pd.DataFrame({"open": close, "high": np.array(close) + 1,
                         "low": np.array(close) - 1, "close": close,
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"forecast": 0.0, "forecast_rank": 0.5,
                        "atr": 1.0}, index=index)
    ind.iloc[offset:offset + 2, 0] = 0.02
    ind.iloc[offset:offset + 2, 1] = 0.95
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[offset:].tolist() == [0, 0, 1, 1, 1, 1, 0]
    ind.iloc[offset:offset + 2, 0] = -0.02
    ind.iloc[offset:offset + 2, 1] = 0.05
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[offset:offset + 5].tolist() == [0, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
