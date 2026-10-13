from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    short_window: int = 5
    baseline_window: int = 20
    wick_window: int = 5
    compression_threshold: float = 0.60
    wick_threshold: float = 0.18
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1791261627"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(params.baseline_window + params.short_window + 1, params.wick_window))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if min(params.short_window, params.baseline_window) < 2 or params.wick_window < 1:
            raise ValueError("Invalid lookback lengths")
        if not 0.0 < params.compression_threshold < 1.0 or not 0.0 <= params.wick_threshold < 1.0:
            raise ValueError("Invalid entry thresholds")
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["short_vol"] = out["returns"].rolling(params.short_window, min_periods=params.short_window).std(ddof=0)
        out["baseline_vol"] = out["returns"].rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0).shift(params.short_window)
        out["compression"] = out["short_vol"] / out["baseline_vol"].where(out["baseline_vol"] > 0.0)
        out["bar_range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["lower_wick"] = (data[["open", "close"]].min(axis=1, skipna=False) - data["low"]).clip(lower=0.0)
        out["upper_wick"] = (data["high"] - data[["open", "close"]].max(axis=1, skipna=False)).clip(lower=0.0)
        out["wick_balance"] = out["lower_wick"] - out["upper_wick"]
        out["range_sum"] = out["bar_range"].rolling(params.wick_window, min_periods=params.wick_window).sum()
        out["wick_pressure"] = out["wick_balance"].rolling(params.wick_window, min_periods=params.wick_window).sum() / out["range_sum"].where(out["range_sum"] > 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        # Primitive one: quieter close returns with a modest positive release.
        out["return_agrees"] = ((out["compression"] <= params.compression_threshold) & (out["returns"] > 0.0) & (out["returns"] <= out["baseline_vol"])).fillna(False)
        # Primitive two: cumulative intraday downside rejection.
        out["wick_agrees"] = (out["wick_pressure"] >= params.wick_threshold).fillna(False)
        out["entry"] = out["return_agrees"] & out["wick_agrees"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("Invalid exit parameters")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if holding:
                # Capture the actual next-bar open only once that bar exists.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    holding = False
                elif closes[i] >= entry_price * (1.0 + params.profit_target) or age >= params.max_hold_bars:
                    holding = False
                if not holding:
                    entry_price = np.nan
                    age = 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                holding = True
                entry_price = np.nan
                age = 0
            signals[i] = int(holding)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": [100.0, 100.0, 103.0, 103.0, 103.0, 103.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    target = strategy.generate_signals(data, indicators, None, ShockwaveParams(profit_target=0.02, max_hold_bars=8)).data
    assert target["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    timed = strategy.generate_signals(data, indicators, None, ShockwaveParams(profit_target=0.02, max_hold_bars=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (timed["size"] > 0.0).all()
    assert strategy.warmup_bars(ShockwaveParams()) == 26


if __name__ == "__main__":
    _self_check()
