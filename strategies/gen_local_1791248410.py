from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeasonalityParams:
    season_window: int = 252
    pressure_window: int = 8
    percentile_window: int = 126
    entry_quantile: float = 0.70


class GeneratedStrategy(BaseStrategy[GapSeasonalityParams]):
    strategy_id = "gen_local_1791248410"

    @classmethod
    def params_type(cls) -> type[GapSeasonalityParams]:
        return GapSeasonalityParams

    def warmup_bars(self, params: GapSeasonalityParams) -> int:
        return int(params.season_window + params.pressure_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapSeasonalityParams) -> pd.DataFrame:
        if min(params.season_window, params.pressure_window, params.percentile_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must lie strictly between zero and one")

        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda x: x > 0)
        out["gap"] = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        weekday = data.index.dayofweek
        out["weekday"] = weekday
        out["seasonal_gap"] = np.nan

        for day in range(7):
            mask = pd.Series(weekday == day, index=data.index)
            valid = mask & out["gap"].notna()
            count = valid.astype(float).rolling(params.season_window, min_periods=params.season_window).sum().shift(1)
            total = out["gap"].where(valid, 0.0).rolling(params.season_window, min_periods=params.season_window).sum().shift(1)
            baseline = total / count.where(count > 0)
            out.loc[mask, "seasonal_gap"] = baseline.loc[mask]

        out["gap_surplus"] = out["gap"] - out["seasonal_gap"]
        out["pressure"] = out["gap_surplus"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["threshold"] = out["pressure"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_quantile).shift(1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapSeasonalityParams) -> SignalFrame:
        valid = indicators[["pressure", "threshold"]].notna().all(axis=1)
        entry_condition = valid & indicators["pressure"].gt(indicators["threshold"])
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapSeasonalityParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2018-01-01", periods=900)
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.0003 * t)
    gaps = 0.002 * np.sin(t / 11.0) + 0.0002 * index.dayofweek.to_numpy()
    opening = np.r_[close[0], close[:-1]] * (1.0 + gaps)
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.01, "low": np.minimum(opening, close) * 0.99, "close": close, "volume": 1000000.0}, index=index)
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    expected = (derived[["pressure", "threshold"]].notna().all(axis=1) & derived["pressure"].gt(derived["threshold"])).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert set(result["signal"].unique()) == {0, 1}
    assert result["size"].gt(0).all()
    assert result["signal"].diff().eq(-1).any()
    assert result.iloc[:strategy.warmup_bars(params) - 2]["signal"].eq(0).all()
