from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeasonalityParams:
    seasonal_window: int = 130
    pressure_window: int = 8
    reversal_threshold: float = 0.2

    def __post_init__(self):
        if self.seasonal_window < 20 or self.pressure_window < 1:
            raise ValueError("Invalid rolling windows")
        if not np.isfinite(self.reversal_threshold) or self.reversal_threshold <= 0:
            raise ValueError("reversal_threshold must be finite and positive")


class GeneratedStrategy(BaseStrategy[GapSeasonalityParams]):
    strategy_id = "gen_local_1790883393"

    @classmethod
    def params_type(cls):
        return GapSeasonalityParams

    def warmup_bars(self, params: GapSeasonalityParams) -> int:
        return int(params.seasonal_window + params.pressure_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapSeasonalityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda x: x > 0)
        out["prior_close"] = prior_close
        out["weekday"] = data.index.dayofweek
        out["gap"] = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["close_return"] = (data["close"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["retained_gap"] = np.sign(out["gap"]) * np.minimum(
            out["gap"].abs(),
            (np.sign(out["gap"]) * out["close_return"]).clip(lower=0.0),
        )
        out["seasonal_mean"] = np.nan
        out["seasonal_scale"] = np.nan
        for day in range(5):
            mask = out["weekday"].eq(day)
            valid = mask & out["retained_gap"].notna() & out["gap"].notna()
            count = valid.astype(float).shift(1).rolling(params.seasonal_window).sum()
            count = count.where(count > 0)
            mean = out["retained_gap"].where(valid, 0.0).shift(1).rolling(params.seasonal_window).sum().div(count)
            scale = out["gap"].abs().where(valid, 0.0).shift(1).rolling(params.seasonal_window).sum().div(count)
            out[f"weekday_{day}_count"] = count
            out[f"weekday_{day}_mean"] = mean
            out[f"weekday_{day}_scale"] = scale
            out.loc[mask, "seasonal_mean"] = mean.loc[mask]
            out.loc[mask, "seasonal_scale"] = scale.loc[mask]
        out["gap_surprise"] = (
            (out["retained_gap"] - out["seasonal_mean"])
            / out["seasonal_scale"].clip(lower=1e-6)
        ).replace([np.inf, -np.inf], np.nan)
        out["pressure"] = out["gap_surprise"].rolling(params.pressure_window).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSeasonalityParams,
    ) -> SignalFrame:
        pressure = indicators["pressure"].replace([np.inf, -np.inf], np.nan)
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[pressure.ge(params.reversal_threshold)] = 1.0
        state.loc[pressure.le(-params.reversal_threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapSeasonalityParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({"pressure": [np.nan, 0.3, 0.1, -0.3, -0.1, 0.3, np.nan, -0.3]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 139
