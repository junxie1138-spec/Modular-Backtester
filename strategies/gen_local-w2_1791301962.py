from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_d: float = 0.45
    frac_window: int = 20
    rank_window: int = 126
    vol_window: int = 20
    percentile: float = 0.80
    target_vol: float = 0.12
    profit_target: float = 0.035
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791301962"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @classmethod
    def warmup_bars(cls, params: StrategyParams) -> int:
        return params.frac_window * 2 + params.rank_window + params.vol_window

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        weights = np.empty(params.frac_window)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_d - k + 1) / k

        fd = pd.Series(np.nan, index=data.index, dtype=float)
        fd.iloc[params.frac_window - 1:] = np.convolve(
            (close / close.iloc[0]).to_numpy(), weights, mode="valid"
        )
        trend = fd - fd.ewm(
            span=params.frac_window, min_periods=params.frac_window, adjust=False
        ).mean()
        history = trend.rolling(params.rank_window, min_periods=params.rank_window)
        volatility = close.pct_change().rolling(
            params.vol_window, min_periods=params.vol_window
        ).std() * np.sqrt(252)

        return pd.DataFrame({
            "trend": trend,
            "upper": history.quantile(params.percentile).shift(1),
            "lower": history.quantile(1 - params.percentile).shift(1),
            "three_day_return": close.pct_change(3),
            "volatility": volatility,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        long_entry = (
            (indicators["trend"] > indicators["upper"])
            & (indicators["three_day_return"] < 0)
        ).fillna(False).to_numpy()
        short_entry = (
            (indicators["trend"] < indicators["lower"])
            & (indicators["three_day_return"] > 0)
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        pending = 0
        entry_price = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending:
                side = pending
                entry_price = opens[i]
                entry_bar = i
                pending = 0
            if side:
                gain = side * (closes[i] / entry_price - 1)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    side = 0
            else:
                if long_entry[i]:
                    side = 1
                elif short_entry[i]:
                    side = -1
                pending = side
            raw[i] = side

        size = (params.target_vol / indicators["volatility"].replace(0, np.nan))
        size = size.clip(lower=0.05, upper=0.95).fillna(0.05)
        df = pd.DataFrame({"signal": raw, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
