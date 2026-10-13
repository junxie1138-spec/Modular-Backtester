from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    slope_window: int = 4
    season_window: int = 252
    vol_window: int = 20
    horizon: int = 4
    ridge: float = 1.0


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = "gen_local_1790918966"

    @classmethod
    def params_type(cls) -> type[SeasonalSlopeParams]:
        return SeasonalSlopeParams

    @staticmethod
    def warmup_bars(params: SeasonalSlopeParams) -> int:
        return int(max(2 * params.slope_window + 1, params.vol_window + 1)
                   + params.horizon + params.season_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalSlopeParams) -> pd.DataFrame:
        if (params.slope_window < 2 or params.vol_window < 2
                or params.season_window < 100 or not 3 <= params.horizon <= 5
                or not np.isfinite(params.ridge) or params.ridge <= 0):
            raise ValueError("Invalid slope, seasonal, volatility, horizon or ridge parameter")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.vol_window).std().replace(0, np.nan)
        out["slope"] = out["return"].rolling(params.slope_window).mean()
        out["innovation"] = ((out["slope"] - out["slope"].shift(params.slope_window))
                             / out["volatility"]).replace([np.inf, -np.inf], np.nan)
        out["weekday"] = data.index.dayofweek
        out["origin_weekday"] = out["weekday"].shift(params.horizon)
        out["matured_innovation"] = out["innovation"].shift(params.horizon)
        out["matured_return"] = out["log_close"].diff(params.horizon)
        out["forecast"] = np.nan
        x = out["matured_innovation"]
        y = out["matured_return"]
        valid = x.notna() & y.notna()
        for weekday in range(5):
            prefix = f"weekday_{weekday}_"
            observed = valid & out["origin_weekday"].eq(weekday)
            count = observed.astype(float).rolling(params.season_window).sum()
            denominator = count.replace(0, np.nan)
            out[prefix + "count"] = count
            out[prefix + "mean_x"] = x.where(observed, 0.0).rolling(params.season_window).sum() / denominator
            out[prefix + "mean_y"] = y.where(observed, 0.0).rolling(params.season_window).sum() / denominator
            out[prefix + "mean_xx"] = (x * x).where(observed, 0.0).rolling(params.season_window).sum() / denominator
            out[prefix + "mean_xy"] = (x * y).where(observed, 0.0).rolling(params.season_window).sum() / denominator
            mx = out[prefix + "mean_x"]
            my = out[prefix + "mean_y"]
            out[prefix + "variance"] = (out[prefix + "mean_xx"] - mx * mx).clip(lower=0)
            out[prefix + "covariance"] = out[prefix + "mean_xy"] - mx * my
            out[prefix + "beta"] = out[prefix + "covariance"] / (out[prefix + "variance"] + params.ridge)
            out[prefix + "prediction"] = my + out[prefix + "beta"] * (out["innovation"] - mx)
            selected = out["weekday"].eq(weekday) & count.ge(20)
            out.loc[selected, "forecast"] = out.loc[selected, prefix + "prediction"]
        out.loc[out.index[:GeneratedStrategy.warmup_bars(params)], "forecast"] = np.nan
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalSlopeParams) -> SignalFrame:
        entry_condition = indicators["forecast"].fillna(0.0).gt(0.0)
        # A false condition exits immediately; re-entry needs two consecutive true bars.
        confirmed = entry_condition & entry_condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame(index=index)
    forecasts = pd.DataFrame({"forecast": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, forecasts, None, SeasonalSlopeParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
