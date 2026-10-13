from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    range_window: int = 14
    baseline_window: int = 63
    correlation_window: int = 6
    compression_threshold: float = 1.1
    prior_correlation_max: float = -0.25
    release_correlation_min: float = 0.1
    profit_target: float = 0.04
    max_hold_bars: int = 20

    def __post_init__(self):
        if min(self.range_window, self.baseline_window, self.correlation_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < self.profit_target < 1 and self.max_hold_bars > 0 and self.compression_threshold > 0):
            raise ValueError("Invalid target, holding period, or compression threshold.")
        if not (-1 <= self.prior_correlation_max < self.release_correlation_min <= 1):
            raise ValueError("Invalid correlation thresholds.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791312658"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(200, params.range_window + 1, params.baseline_window + 2, 2 * params.correlation_window + 2))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["close"] = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        close = out["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["baseline_vol"] = out["return"].rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        out["range_high"] = close.rolling(params.range_window, min_periods=params.range_window).max()
        out["range_low"] = close.rolling(params.range_window, min_periods=params.range_window).min()
        out["compression"] = (out["range_high"] - out["range_low"]) / close / (out["baseline_vol"].replace(0, np.nan) * np.sqrt(params.range_window))
        out["prior_compression"] = out["compression"].shift(1)
        out["correlation"] = out["return"].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out["lagged_return"]).clip(-1, 1)
        out["prior_correlation"] = out["correlation"].shift(params.correlation_window)
        out = out.replace([np.inf, -np.inf], np.nan)
        required = ["close", "return", "ma200", "prior_compression", "correlation", "prior_correlation"]
        out["ready"] = out[required].notna().all(axis=1)
        out["release"] = out["ready"] & (out["prior_compression"] <= params.compression_threshold) & (out["prior_correlation"] <= params.prior_correlation_max) & (out["correlation"] >= params.release_correlation_min)
        out["long_entry"] = out["release"] & (close > out["ma200"]) & (out["return"] > 0)
        out["short_entry"] = out["release"] & (close < out["ma200"]) & (out["return"] < 0)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The preceding close's entry decision fills at this bar's open.
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                age = i - entry_bar + 1
                if (np.isfinite(gain) and gain >= params.profit_target) or age >= params.max_hold_bars:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
                    continue
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(max_hold_bars=15)
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({name: 100.0 for name in ["open", "high", "low", "close", "volume"]}, index=index)
    flags = pd.DataFrame({"long_entry": False, "short_entry": False}, index=index)
    flags.iloc[0, 0] = True
    timed = strategy.generate_signals(data, flags, None, params).data
    assert timed["signal"].iloc[0] == 0
    assert (timed["signal"].iloc[1:16] == 1).all() and timed["signal"].iloc[16] == 0
    data.iloc[1, data.columns.get_loc("close")] = 105.0
    targeted = strategy.generate_signals(data, flags, None, params).data
    assert targeted["signal"].tolist()[:3] == [0, 1, 0]
    flags.iloc[0] = [False, True]
    data.iloc[1, data.columns.get_loc("close")] = 95.0
    short = strategy.generate_signals(data, flags, None, params).data
    assert short["signal"].tolist()[:3] == [0, -1, 0]
    flat = pd.DataFrame({name: 100.0 for name in data.columns}, index=pd.date_range("2020-01-01", periods=240))
    derived = strategy.indicators(flat, params)
    assert not derived["long_entry"].any() and not derived["short_entry"].any()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
