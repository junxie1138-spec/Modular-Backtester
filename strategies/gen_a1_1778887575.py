from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    window: int = 20
    tide_lookback: int = 5
    vol_window: int = 20
    vol_mult: float = 1.3
    profit_target_pct: float = 0.05
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_a1_1778887575"

    @classmethod
    def params_type(cls):
        return TideParams

    @staticmethod
    def warmup_bars(params: TideParams) -> int:
        return int(max(params.window + params.tide_lookback, params.vol_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        w = max(int(params.window), 2)
        k = max(int(params.tide_lookback), 1)
        vw = max(int(params.vol_window), 2)

        # High-water mark: the rolling tide envelope.
        hwm = close.rolling(w, min_periods=w).max()

        # Trend-strength primitive: the tide line is advancing.
        tide_rising = (hwm > hwm.shift(k)).fillna(False)

        # A fresh window high is set when today's close equals the rolling max.
        new_high = (close >= hwm).fillna(False)

        # Volume-confirmed move: the new high prints on above-average volume.
        vol_ma = volume.rolling(vw, min_periods=vw).mean()
        vol_ratio = (volume / vol_ma).replace([np.inf, -np.inf], np.nan)
        vol_confirmed = (vol_ratio > float(params.vol_mult)).fillna(False)

        # Two-primitive AND: rising tide AND volume-confirmed fresh high.
        raw_entry = (tide_rising & new_high & vol_confirmed).astype(int)

        out = pd.DataFrame(index=data.index)
        out["hwm"] = hwm
        out["tide_rising"] = tide_rising.astype(int)
        out["new_high"] = new_high.astype(int)
        out["vol_ratio"] = vol_ratio.fillna(0.0)
        out["vol_confirmed"] = vol_confirmed.astype(int)
        out["raw_entry"] = raw_entry
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        raw = indicators["raw_entry"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)

        target = float(params.profit_target_pct)
        max_hold = max(int(params.max_hold_bars), 1)

        position = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if position == 1:
                bars_held += 1
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                # Exit: profit-target OR time-stop, whichever fires first.
                if gain >= target or bars_held >= max_hold:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if raw[i] >= 1.0 and np.isfinite(close[i]) and close[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
