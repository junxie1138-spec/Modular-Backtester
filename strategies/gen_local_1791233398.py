from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRangeParams:
    seasonal_lookback: int = 756
    minimum_month_samples: int = 30
    seasonal_threshold: float = 0.00015
    range_lookback: int = 5
    pressure_threshold: float = 0.08
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SeasonalRangeParams]):
    strategy_id = "gen_local_1791233398"

    @classmethod
    def params_type(cls) -> type[SeasonalRangeParams]:
        return SeasonalRangeParams

    def warmup_bars(self, params: SeasonalRangeParams) -> int:
        return int(max(params.seasonal_lookback + 1, params.range_lookback + 1))

    def indicators(self, data: pd.DataFrame, params: SeasonalRangeParams) -> pd.DataFrame:
        if min(params.seasonal_lookback, params.minimum_month_samples, params.range_lookback, params.holding_bars) < 1:
            raise ValueError("Lookbacks, sample count, and holding bars must be positive")
        if params.seasonal_threshold < 0 or params.pressure_threshold < 0:
            raise ValueError("Thresholds must be nonnegative")

        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["upper_extension"] = data["high"].diff().clip(lower=0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0)
        out["upper_extension_sum"] = out["upper_extension"].rolling(params.range_lookback, min_periods=params.range_lookback).sum()
        out["lower_extension_sum"] = out["lower_extension"].rolling(params.range_lookback, min_periods=params.range_lookback).sum()
        out["range_sum"] = out["daily_range"].rolling(params.range_lookback, min_periods=params.range_lookback).sum()
        out["range_pressure"] = ((out["upper_extension_sum"] - out["lower_extension_sum"]) / out["range_sum"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["intraday_return"] = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["seasonal_mean"] = np.nan
        out["seasonal_sample_count"] = np.nan
        months = pd.Series(data.index.month, index=data.index)
        for month in range(1, 13):
            member = months.eq(month)
            valid = member & out["intraday_return"].notna()
            out[f"month_{month}_return_sum"] = out["intraday_return"].where(valid, 0.0).shift(1).rolling(params.seasonal_lookback, min_periods=params.seasonal_lookback).sum()
            out[f"month_{month}_sample_count"] = valid.astype(float).shift(1).rolling(params.seasonal_lookback, min_periods=params.seasonal_lookback).sum()
            count = out[f"month_{month}_sample_count"]
            mean = out[f"month_{month}_return_sum"] / count.where(count >= params.minimum_month_samples)
            out.loc[member, "seasonal_mean"] = mean.loc[member]
            out.loc[member, "seasonal_sample_count"] = count.loc[member]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalRangeParams) -> SignalFrame:
        season = indicators["seasonal_mean"]
        pressure = indicators["range_pressure"]
        long_entry = (season > params.seasonal_threshold) & (pressure > params.pressure_threshold)
        short_entry = (season < -params.seasonal_threshold) & (pressure < -params.pressure_threshold)
        candidates = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar == params.holding_bars:
                    position = 0
                signal[i] = position
                continue
            if candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _check_fixed_exit() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"seasonal_mean": [0.01] + [-0.01] * 8, "range_pressure": [0.5] + [-0.5] * 8}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, SeasonalRangeParams(holding_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _check_fixed_exit()
