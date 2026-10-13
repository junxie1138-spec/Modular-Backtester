from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilParams:
    peak_window: int = 63
    rank_window: int = 126
    fast_window: int = 5
    slow_window: int = 20
    atr_window: int = 14
    direction_window: int = 4
    depth_quantile: float = 0.75
    compression_quantile: float = 0.30
    spike_quantile: float = 0.95
    refractory_bars: int = 5
    hold_bars: int = 18
    stop_atr: float = 2.5
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[CoilParams]):
    strategy_id = "gen_local_1790990514"

    @classmethod
    def params_type(cls) -> type[CoilParams]:
        return CoilParams

    def warmup_bars(self, params: CoilParams) -> int:
        return int(
            max(
                params.peak_window + params.rank_window + 1,
                max(params.fast_window, params.slow_window) + params.rank_window + 1,
                params.rank_window + params.refractory_bars + 2,
                params.peak_window + params.direction_window,
                params.atr_window + 1,
            )
        )

    def indicators(self, data: pd.DataFrame, params: CoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["peak"] = close.rolling(params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        out["depth_threshold"] = out["drawdown"].shift(1).rolling(
            params.rank_window
        ).quantile(params.depth_quantile)
        out["depth_change"] = out["drawdown"].diff(params.direction_window)
        out["relative_range"] = out["true_range"] / close.where(close > 0)
        out["fast_range"] = out["relative_range"].rolling(params.fast_window).mean()
        out["slow_range"] = out["relative_range"].rolling(params.slow_window).mean()
        out["compression"] = out["fast_range"] / out["slow_range"].where(
            out["slow_range"] > 0
        )
        out["compression_threshold"] = out["compression"].shift(1).rolling(
            params.rank_window
        ).quantile(params.compression_quantile)
        out["spike_threshold"] = out["relative_range"].shift(1).rolling(
            params.rank_window
        ).quantile(params.spike_quantile)
        out["spike"] = (
            out["relative_range"] > out["spike_threshold"]
        ).fillna(False).astype(int)
        out["recent_spike"] = out["spike"].rolling(params.refractory_bars).max()
        required = [
            "atr", "drawdown", "depth_threshold", "depth_change",
            "compression", "compression_threshold", "spike_threshold", "recent_spike",
        ]
        out["ready"] = np.isfinite(out[required]).all(axis=1)
        out["eligible"] = (
            out["ready"]
            & (out["atr"] > 0)
            & (out["drawdown"] > out["depth_threshold"])
            & (out["compression"] < out["compression_threshold"])
            & (out["recent_spike"] == 0)
        ).fillna(False)
        out["entry_direction"] = np.select(
            [
                out["eligible"] & (out["depth_change"] < 0),
                out["eligible"] & (out["depth_change"] > 0),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_index = -1
        entry_price = np.nan
        frozen_atr = np.nan
        last_exit = -params.cooldown_bars - 1

        for i in range(len(data)):
            if side != 0:
                # The prior close's decision fills at this session's open.
                if i == entry_index:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                        last_exit = i
                        continue
                # Both directions use a fixed distance from the actual entry.
                stopped = np.isfinite(closes[i]) and (
                    (side == 1 and closes[i] < entry_price - params.stop_atr * frozen_atr)
                    or (side == -1 and closes[i] > entry_price + params.stop_atr * frozen_atr)
                )
                expired = i - entry_index + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    last_exit = i
                    continue
            elif i - last_exit > params.cooldown_bars:
                if entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                    side = int(entries[i])
                    entry_index = i + 1
                    entry_price = np.nan
                    frozen_atr = atr[i]
            signals[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
