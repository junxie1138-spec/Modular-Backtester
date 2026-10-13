from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownEchoParams:
    depth_threshold: float = 0.008
    correlation_threshold: float = 0.25
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[DrawdownEchoParams]):
    strategy_id = "gen_local-w2_1790986969"

    @classmethod
    def params_type(cls) -> type[DrawdownEchoParams]:
        return DrawdownEchoParams

    def warmup_bars(self, params: DrawdownEchoParams) -> int:
        # Three closes for depth, one difference, one lag, six pairs.
        return 10

    def indicators(self, data: pd.DataFrame, params: DrawdownEchoParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        peak = close.rolling(3, min_periods=3).max()
        depth = (1.0 - close / peak).clip(lower=0.0)
        change = depth.diff()
        lagged_change = change.shift(1)
        correlation = change.rolling(6, min_periods=6).corr(lagged_change)
        correlation = correlation.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        forecast = correlation * change
        valid = depth.notna() & change.notna() & correlation.notna()
        return pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "depth_change": change,
            "lagged_depth_change": lagged_change,
            "depth_autocorrelation": correlation,
            "forecast_depth_change": forecast,
            "valid": valid,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownEchoParams) -> SignalFrame:
        eligible = (
            indicators["valid"].fillna(False)
            & indicators["depth"].ge(params.depth_threshold)
            & indicators["depth_autocorrelation"].abs().ge(params.correlation_threshold)
        )
        forecast = indicators["forecast_depth_change"].fillna(0.0)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & forecast.lt(0.0)] = 1
        candidates.loc[eligible & forecast.gt(0.0)] = -1

        holding = max(1, int(params.holding_bars))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(candidates.to_numpy()):
            if position != 0:
                if i - entry_bar >= holding:
                    position = 0
                signal[i] = position
                # The expiry bar stays flat; new entries start next bar.
                continue
            if candidate != 0:
                position = int(candidate)
                entry_bar = i
            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
