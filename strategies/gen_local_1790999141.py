from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    lookback: int = 9
    shallow_depth: float = 0.6
    deep_depth: float = 3.2


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790999141"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return min(9, max(3, int(params.lookback))) + 1

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        n = min(9, max(3, int(params.lookback)))
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.div(close.shift(1)).sub(1).replace([np.inf, -np.inf], np.nan)
        peak = close.rolling(n, min_periods=n).max()
        downside_energy = returns.clip(upper=0).pow(2)
        total_energy = returns.pow(2)
        downside_noise = downside_energy.rolling(n - 1, min_periods=n - 1).mean().pow(0.5)
        total_noise = total_energy.rolling(n - 1, min_periods=n - 1).mean().pow(0.5)
        noise_scale = downside_noise.clip(lower=total_noise * 0.25).clip(lower=1e-8)
        drawdown = (1.0 - close.div(peak)).clip(lower=0)
        depth = drawdown.div(noise_scale).replace([np.inf, -np.inf], np.nan)
        valid = depth.notna()
        shallow = max(0.0, float(params.shallow_depth))
        deep = max(shallow + 0.1, float(params.deep_depth))
        bullish = valid & depth.le(shallow)
        bearish = valid & depth.ge(deep)
        confirmed_long = bullish & bullish.shift(1, fill_value=False)
        confirmed_short = bearish & bearish.shift(1, fill_value=False)
        trigger = pd.Series(np.select([confirmed_long, confirmed_short], [1, -1], default=0), index=data.index, dtype=int)
        return pd.DataFrame({
            "returns": returns,
            "peak": peak,
            "downside_energy": downside_energy,
            "total_energy": total_energy,
            "downside_noise": downside_noise,
            "total_noise": total_noise,
            "noise_scale": noise_scale,
            "drawdown": drawdown,
            "normalized_depth": depth,
            "valid": valid,
            "bullish": bullish,
            "bearish": bearish,
            "confirmed_long": confirmed_long,
            "confirmed_short": confirmed_short,
            "trigger": trigger,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        triggers = indicators["trigger"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, trigger in enumerate(triggers):
            if state == 0:
                state = int(trigger)
            elif trigger == -state:
                state = 0
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
