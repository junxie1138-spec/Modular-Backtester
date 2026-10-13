from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCapacityParams:
    energy_window: int = 20
    energy_multiple: float = 1.8
    channel_window: int = 12
    capacity_window: int = 8
    capacity_multiple: float = 0.35
    atr_window: int = 14
    stop_atr: float = 1.7
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletCapacityParams]):
    strategy_id = "gen_local_1790985840"

    @classmethod
    def params_type(cls) -> type[WaveletCapacityParams]:
        return WaveletCapacityParams

    def warmup_bars(self, params: WaveletCapacityParams) -> int:
        return int(max(
            16 + max(1, params.energy_window) + 1,
            max(1, params.channel_window) + 1,
            max(1, params.capacity_window) + 1,
            max(1, params.atr_window) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: WaveletCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for scale in (2, 4, 8):
            block = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (block - block.shift(scale))
        coefficients = out[["haar_2", "haar_4", "haar_8"]]
        out["energy"] = coefficients.pow(2).sum(axis=1, min_count=3)
        energy_window = max(1, params.energy_window)
        out["energy_baseline"] = out["energy"].shift(1).rolling(
            energy_window, min_periods=energy_window
        ).mean()
        out["energy_ratio"] = out["energy"] / out["energy_baseline"].where(
            out["energy_baseline"] > 0
        )
        channel_window = max(1, params.channel_window)
        out["channel_high"] = data["high"].shift(1).rolling(
            channel_window, min_periods=channel_window
        ).max()
        out["bar_range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        body_top = data[["open", "close"]].max(axis=1)
        out["upper_wick_turnover"] = data["volume"].clip(lower=0) * (
            (data["high"] - body_top).clip(lower=0) / out["bar_range"]
        ).clip(0, 1)
        capacity_window = max(1, params.capacity_window)
        out["resistance_capacity"] = out["upper_wick_turnover"].shift(1).rolling(
            capacity_window, min_periods=capacity_window
        ).sum()
        out["bullish_turnover"] = data["volume"].clip(lower=0) * (
            (data["close"] - data["open"]).clip(lower=0) / out["bar_range"]
        ).clip(0, 1)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_window = max(1, params.atr_window)
        out["atr"] = out["true_range"].rolling(
            atr_window, min_periods=atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletCapacityParams,
    ) -> SignalFrame:
        wavelet_agrees = (
            (indicators["haar_2"] > 0)
            & (indicators["haar_4"] > 0)
            & (indicators["haar_8"] > 0)
            & (indicators["energy_ratio"] > params.energy_multiple)
        )
        capacity_agrees = (
            (data["close"] > indicators["channel_high"])
            & (indicators["bullish_turnover"] > 0)
            & (indicators["bullish_turnover"] >
               params.capacity_multiple * indicators["resistance_capacity"])
        )
        required = indicators[[
            "haar_2", "haar_4", "haar_8", "energy_ratio",
            "channel_high", "bullish_turnover", "resistance_capacity", "atr",
        ]]
        ready = required.notna().all(axis=1) & (indicators["atr"] > 0)
        candidate = (wavelet_agrees & capacity_agrees & ready).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        fixed_stop = np.nan
        holding_bars = min(5, max(3, int(params.holding_bars)))
        for i in range(len(data)):
            if pending:
                # The prior close's decision fills at this bar's open.
                # Freeze the entry ATR and stop; neither follows later prices.
                active = bool(np.isfinite(opens[i]) and opens[i] > 0)
                if active:
                    fixed_stop = opens[i] - max(0.01, params.stop_atr) * atr[i - 1]
                    entry_bar = i
                pending = False
            if active:
                stop_hit = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                time_exit = i - entry_bar + 1 >= holding_bars
                if stop_hit or time_exit:
                    active = False
                else:
                    raw[i] = 1
            elif candidate[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
