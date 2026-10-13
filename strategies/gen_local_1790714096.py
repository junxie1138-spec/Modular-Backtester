from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTrendParams:
    trend_days: int = 200
    efficiency_days: int = 20
    atr_days: int = 14
    gap_noise_days: int = 20
    gap_z_min: float = 0.75
    efficiency_min: float = 0.35
    stop_atr: float = 2.5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapTrendParams]):
    strategy_id = "gen_local_1790714096"

    @classmethod
    def params_type(cls):
        return GapTrendParams

    def warmup_bars(self, params: GapTrendParams) -> int:
        return max(params.trend_days, params.efficiency_days + 1, params.atr_days, params.gap_noise_days + 1)

    def indicators(self, data: pd.DataFrame, params: GapTrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["trend_ma"] = data["close"].rolling(params.trend_days, min_periods=params.trend_days).mean()
        daily_return = data["close"].pct_change()
        abs_move = daily_return.abs().rolling(params.efficiency_days, min_periods=params.efficiency_days).sum()
        net_move = (data["close"] / data["close"].shift(params.efficiency_days) - 1).abs()
        out["efficiency"] = net_move / abs_move.replace(0, np.nan)
        gap = (data["open"] / prev_close) - 1
        gap_noise = gap.rolling(params.gap_noise_days, min_periods=params.gap_noise_days).std()
        out["gap_z"] = gap / gap_noise.replace(0, np.nan)
        out["gap"] = gap
        out["trend_up"] = (data["close"] > out["trend_ma"]) & (out["trend_ma"] > out["trend_ma"].shift(20))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTrendParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        valid = indicators[["atr", "trend_ma", "efficiency", "gap_z"]].notna().all(axis=1)
        entry = valid & indicators["trend_up"] & (indicators["efficiency"] >= params.efficiency_min) & (indicators["gap_z"] >= params.gap_z_min) & (indicators["gap"] > 0)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_atr = 0.0
        for i in range(len(data)):
            if active and data["close"].iloc[i] < entry_price - params.stop_atr * entry_atr:
                active = False
            elif not active and bool(entry.iloc[i]):
                active = True
                entry_price = float(data["close"].iloc[i])
                entry_atr = float(indicators["atr"].iloc[i])
            signal[i] = int(active)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
