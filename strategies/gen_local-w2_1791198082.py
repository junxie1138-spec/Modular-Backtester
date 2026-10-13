from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CounterGapParams:
    ma_window: int = 20
    breakout_window: int = 4
    z_threshold: float = 1.2
    profit_target: float = 0.007
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CounterGapParams]):
    strategy_id = "gen_local-w2_1791198082"

    @classmethod
    def params_type(cls) -> type[CounterGapParams]:
        return CounterGapParams

    def warmup_bars(self, params: CounterGapParams) -> int:
        return int(max(params.ma_window, params.breakout_window + 1, 2))

    def indicators(self, data: pd.DataFrame, params: CounterGapParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.breakout_window < 1:
            raise ValueError("Lookback windows must be positive; ma_window must be at least two.")
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")
        if params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        out["prior_high"] = data["high"].rolling(params.breakout_window).max().shift(1)
        out["prior_low"] = data["low"].rolling(params.breakout_window).min().shift(1)
        out["previous_close"] = close.shift(1)
        out["gap"] = data["open"] / out["previous_close"].replace(0.0, np.nan) - 1.0
        out["body"] = close - data["open"]
        out["range"] = data["high"] - data["low"]
        out["body_fraction"] = (out["body"].abs() / out["range"].replace(0.0, np.nan)).clip(0.0, 1.0)
        out["conviction"] = (0.25 + 0.25 * out["z"].abs() / params.z_threshold + 0.15 * out["body_fraction"]).clip(0.25, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CounterGapParams) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        long_entry = valid & (indicators["z"] >= params.z_threshold) & (indicators["gap"] < 0.0) & (indicators["body"] > 0.0) & (data["close"] > indicators["prior_high"])
        short_entry = valid & (indicators["z"] <= -params.z_threshold) & (indicators["gap"] > 0.0) & (indicators["body"] < 0.0) & (data["close"] < indicators["prior_low"])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        conviction = indicators["conviction"].fillna(0.25).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        active_size = 1.0
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    held_bars = i - entry_bar + 1
                    if gain >= params.profit_target or held_bars >= params.max_hold_bars:
                        position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                active_size = float(conviction[i])
            signals[i] = position
            sizes[i] = active_size if position != 0 else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
