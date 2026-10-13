from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCurvatureParams:
    curve_window: int = 3
    vol_window: int = 10
    wick_threshold: float = 0.10
    target_vol: float = 0.012


class GeneratedStrategy(BaseStrategy[RangeCurvatureParams]):
    strategy_id = "gen_local_1791295614"

    @classmethod
    def params_type(cls) -> type[RangeCurvatureParams]:
        return RangeCurvatureParams

    @classmethod
    def warmup_bars(cls, params: RangeCurvatureParams) -> int:
        return max(params.curve_window + 2, params.vol_window)

    def indicators(self, data: pd.DataFrame, params: RangeCurvatureParams) -> pd.DataFrame:
        span = (data["high"] - data["low"]).clip(lower=1e-12)
        midpoint = (data["high"] + data["low"]) / 2
        curvature = midpoint.diff().diff().rolling(params.curve_window).mean() / span
        lower_wick = (data[["open", "close"]].min(axis=1) - data["low"]).clip(lower=0)
        upper_wick = (data["high"] - data[["open", "close"]].max(axis=1)).clip(lower=0)
        wick_balance = ((lower_wick - upper_wick) / span).rolling(params.curve_window).mean()
        log_range = np.log(data["high"].clip(lower=1e-12) / data["low"].clip(lower=1e-12))
        volatility = np.sqrt(log_range.pow(2).rolling(params.vol_window).mean() / (4 * np.log(2)))
        return pd.DataFrame({
            "curvature": curvature.fillna(0.0),
            "wick_balance": wick_balance.fillna(0.0),
            "volatility": volatility.fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeCurvatureParams,
    ) -> SignalFrame:
        long_entry = (indicators["curvature"] > 0) & (indicators["wick_balance"] > params.wick_threshold)
        short_entry = (indicators["curvature"] < 0) & (indicators["wick_balance"] < -params.wick_threshold)
        positions = []
        position = 0
        for go_long, go_short in zip(long_entry, short_entry):
            if position == 0:
                position = 1 if go_long else -1 if go_short else 0
            elif position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            positions.append(position)

        size = (params.target_vol / indicators["volatility"].replace(0, np.nan)).fillna(1.0).clip(0.1, 1.0)
        df = pd.DataFrame({"signal": positions, "size": size.astype(float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
