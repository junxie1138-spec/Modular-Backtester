from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    window: int = 8
    strength: float = 0.35
    premium: float = 0.08


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1791339292"

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        if params.window < 2 or not 0 < params.strength < 1 or not 0 < params.premium < 2:
            raise ValueError("Require window >= 2, 0 < strength < 1, and 0 < premium < 2")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["absolute_return"] = out["return"].abs()
        out["volume"] = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan).clip(lower=0)
        out["weighted_return"] = out["return"] * out["volume"]
        out["weighted_absolute_return"] = out["absolute_return"] * out["volume"]
        out["net"] = out["return"].rolling(params.window, min_periods=params.window).sum()
        out["gross"] = out["absolute_return"].rolling(params.window, min_periods=params.window).sum()
        out["weighted_net"] = out["weighted_return"].rolling(params.window, min_periods=params.window).sum()
        out["weighted_gross"] = out["weighted_absolute_return"].rolling(params.window, min_periods=params.window).sum()
        out["efficiency"] = out["net"] / out["gross"].replace(0, np.nan)
        out["volume_efficiency"] = out["weighted_net"] / out["weighted_gross"].replace(0, np.nan)
        out["participation_premium"] = out["volume_efficiency"] - out["efficiency"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        entry = indicators["volume_efficiency"].ge(params.strength) & indicators["participation_premium"].ge(params.premium)
        reversal = indicators["volume_efficiency"].le(-params.strength) & indicators["participation_premium"].le(-params.premium)
        df = pd.DataFrame(index=data.index)
        df["signal"] = np.nan
        df.loc[entry, "signal"] = 1
        df.loc[reversal, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "volume_efficiency": [np.nan, 0.6, 0.1, -0.6, -0.1, 0.6],
        "participation_premium": [np.nan, 0.2, 0.01, -0.2, -0.01, 0.2],
    }, index=index)
    strategy = GeneratedStrategy()
    params = ParticipationParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
