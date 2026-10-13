from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    roc_bars: int = 3
    vol_bars: int = 20
    hold_bars: int = 8
    accel_floor: float = 0.004
    shock_multiple: float = 1.7
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791298971"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return max(2 * params.roc_bars + 1, params.vol_bars + 2)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        roc = data["close"].pct_change(params.roc_bars)
        acceleration = roc - roc.shift(params.roc_bars)
        daily_range = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
        prior_range = daily_range.shift(1)
        baseline = daily_range.rolling(params.vol_bars).median().shift(2)
        damping = (prior_range >= params.shock_multiple * baseline) & (daily_range <= 0.7 * prior_range)
        realized_vol = data["close"].pct_change().rolling(params.vol_bars).std() * np.sqrt(252.0)
        return pd.DataFrame({
            "acceleration": acceleration.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "damping": damping.fillna(False),
            "realized_vol": realized_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        entry = (indicators["acceleration"] > params.accel_floor) & indicators["damping"]
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, should_enter in enumerate(entry.to_numpy(dtype=bool)):
            if remaining:
                remaining -= 1
                signal[i] = int(remaining > 0)
            elif should_enter:
                remaining = params.hold_bars
                signal[i] = 1
        size = (params.target_vol / indicators["realized_vol"].replace(0, np.nan)).clip(0.05, 0.95).fillna(0.05)
        df = pd.DataFrame({"signal": signal, "size": size.astype(float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
