from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_lookback: int = 25
    trough_lookback: int = 5
    volume_lookback: int = 20
    min_drawdown: float = 0.03
    recovery_fraction: float = 0.4
    volume_multiple: float = 1.3
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791226746"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.peak_lookback, params.trough_lookback, params.volume_lookback) + 2

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        peak = data["high"].rolling(params.peak_lookback).max().shift(1)
        trough = data["low"].rolling(params.trough_lookback).min().shift(1)
        median_volume = data["volume"].rolling(params.volume_lookback).median().shift(1)
        out = pd.DataFrame(index=data.index)
        out["drawdown"] = ((peak - trough) / peak.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["recovery"] = ((data["close"] - trough) / (peak - trough).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["relative_volume"] = (data["volume"] / median_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        recovery_event = (
            indicators["drawdown"].ge(params.min_drawdown)
            & indicators["recovery"].ge(params.recovery_fraction)
            & indicators["recovery"].shift(1).lt(params.recovery_fraction)
        )
        volume_move = (
            data["close"].gt(data["open"])
            & data["close"].gt(data["close"].shift(1))
            & indicators["relative_volume"].ge(params.volume_multiple)
        )
        entry = (recovery_event & volume_move).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy()
        signals = []
        holding = False
        entry_bar = 0
        entry_close = 0.0
        for i, close in enumerate(closes):
            if holding:
                if close >= entry_close * (1.0 + params.profit_target) or i - entry_bar >= params.hold_bars:
                    holding = False
                signals.append(int(holding))
            elif i >= self.warmup_bars(params) and entry[i]:
                holding = True
                entry_bar = i
                entry_close = float(close)
                signals.append(1)
            else:
                signals.append(0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
