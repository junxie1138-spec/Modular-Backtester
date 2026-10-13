from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    ma_window: int = 20
    compression_window: int = 5
    baseline_window: int = 40
    atr_window: int = 14
    compression_limit: float = 0.75
    entry_z: float = 1.25
    stop_atr: float = 1.5
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1790987380"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.ma_window + 1, params.compression_window + 2,
                       params.baseline_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # Compression is established before the candidate session.
        out["fast_range"] = out["true_range"].rolling(params.compression_window).mean().shift(1)
        out["baseline_range"] = out["true_range"].rolling(params.baseline_window).mean().shift(1)
        out["compression"] = out["fast_range"] / out["baseline_range"].replace(0.0, np.nan)
        out["ma"] = data["close"].rolling(params.ma_window).mean().shift(1)
        out["sigma"] = data["close"].rolling(params.ma_window).std(ddof=0).shift(1)
        out["z"] = (data["close"] - out["ma"]) / out["sigma"].replace(0.0, np.nan)
        out["intraday_rejection"] = (data["close"] - data["open"]) / out["atr"].replace(0.0, np.nan)
        out["range_ratio"] = out["true_range"] / out["baseline_range"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        rejection = indicators["intraday_rejection"].to_numpy(dtype=float)
        compression = indicators["compression"].to_numpy(dtype=float)
        range_ratio = indicators["range_ratio"].to_numpy(dtype=float)
        position = 0
        entry_pending = False
        entry_price = 0.0
        entry_atr = 0.0
        age = 0
        exposure = 1.0
        threshold = max(float(params.entry_z), 0.01)
        stop_multiple = max(float(params.stop_atr), 0.01)
        holding = min(5, max(3, int(params.hold_bars)))
        lower_size = min(1.0, max(0.01, float(params.min_size)))
        upper_size = min(1.0, max(lower_size, float(params.max_size)))

        for i in range(n):
            if position:
                if entry_pending:
                    # The preceding close's order fills at this session's open.
                    # Entry ATR was frozen when that order was decided.
                    entry_price = opening[i]
                    entry_pending = False
                age += 1
                valid_prices = np.isfinite(entry_price) and np.isfinite(close[i]) and entry_price > 0.0
                stop_hit = valid_prices and (
                    (position == 1 and close[i] < entry_price - stop_multiple * entry_atr)
                    or (position == -1 and close[i] > entry_price + stop_multiple * entry_atr)
                )
                if not valid_prices or stop_hit or age >= holding:
                    position = 0
                    exposure = 1.0
                    age = 0
            elif i >= self.warmup_bars(params) - 1:
                valid = (
                    np.isfinite(close[i]) and close[i] > 0.0
                    and np.isfinite(atr[i]) and atr[i] > 0.0
                    and np.isfinite(z[i]) and np.isfinite(rejection[i])
                    and np.isfinite(compression[i]) and np.isfinite(range_ratio[i])
                )
                compressed = valid and compression[i] <= params.compression_limit and range_ratio[i] <= 1.0
                # A stretched close must meet opposite intraday pressure:
                # positive open-to-close return below the MA, or vice versa.
                if compressed and abs(z[i]) >= threshold and z[i] * rejection[i] < 0.0:
                    position = 1 if z[i] < 0.0 else -1
                    strength = float(np.clip(abs(z[i]) / threshold - 1.0, 0.0, 1.0))
                    exposure = lower_size + (upper_size - lower_size) * strength
                    entry_atr = atr[i]
                    entry_pending = True
                    age = 0
            signals[i] = position
            sizes[i] = exposure

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
