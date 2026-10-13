from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityBridgeParams:
    vol_lookback: int = 20
    min_bridge_score: float = 1.0
    target_vol: float = 0.12
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityBridgeParams]):
    strategy_id = "gen_local_1791300303"

    @classmethod
    def params_type(cls) -> type[VisibilityBridgeParams]:
        return VisibilityBridgeParams

    def warmup_bars(self, params: VisibilityBridgeParams) -> int:
        return max(params.vol_lookback + 1, 7)

    def indicators(self, data: pd.DataFrame, params: VisibilityBridgeParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        daily_vol = np.log(close).diff().rolling(params.vol_lookback).std(ddof=0)
        daily_vol = daily_vol.replace(0, np.nan)
        tension = pd.Series(0.0, index=data.index)
        for span in range(3, 7):
            interior_high = close.shift(1).rolling(span - 1).max()
            clearance = (np.minimum(close, close.shift(span)) - interior_high).clip(lower=0).fillna(0.0)
            tension = tension + (span - 2) * clearance
        score = (tension / (close * daily_vol)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        annual_vol = (daily_vol * np.sqrt(252)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({"bridge_score": score, "annual_vol": annual_vol}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityBridgeParams,
    ) -> SignalFrame:
        score = indicators["bridge_score"].to_numpy()
        vol = indicators["annual_vol"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        exit_at = -1
        active_size = 1.0
        threshold = max(params.min_bridge_score, 1e-9)
        for i in range(len(data)):
            if i == exit_at:
                exit_at = -1
                continue
            if i < exit_at:
                signal[i] = 1
                size[i] = active_size
                continue
            if score[i] >= threshold and vol[i] > 0:
                active_size = float(np.clip(params.target_vol / vol[i] * min(score[i] / threshold, 1.5), 0.01, 0.95))
                signal[i] = 1
                size[i] = active_size
                exit_at = i + max(1, params.hold_bars)
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
