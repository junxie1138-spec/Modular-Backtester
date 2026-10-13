from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpirationDiscountParams:
    ma_window: int = 40
    entry_z: float = 0.35
    sizing_span: float = 1.5


class GeneratedStrategy(BaseStrategy[ExpirationDiscountParams]):
    strategy_id = "gen_local-w2_1791240894"

    @classmethod
    def params_type(cls) -> type[ExpirationDiscountParams]:
        return ExpirationDiscountParams

    def warmup_bars(self, params: ExpirationDiscountParams) -> int:
        return max(2, int(params.ma_window))

    def indicators(self, data: pd.DataFrame, params: ExpirationDiscountParams) -> pd.DataFrame:
        window = self.warmup_bars(params)
        close = data["close"].astype(float)
        mean = close.rolling(window, min_periods=window).mean()
        deviation = close.rolling(window, min_periods=window).std(ddof=0)
        z = ((close - mean) / deviation.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        dates = pd.DatetimeIndex(data.index)
        first_weekday = (dates.dayofweek - (dates.day - 1)) % 7
        third_friday = 15 + (4 - first_weekday) % 7
        seasonal_window = (dates.month % 3 == 0) & (dates.day >= third_friday)
        return pd.DataFrame({
            "ma": mean,
            "price_std": deviation,
            "z": z,
            "third_friday_day": third_friday,
            "seasonal_window": seasonal_window,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExpirationDiscountParams) -> SignalFrame:
        z = indicators["z"].replace([np.inf, -np.inf], np.nan)
        threshold = max(0.0, float(params.entry_z))
        span = max(1e-12, float(params.sizing_span))
        eligible = indicators["seasonal_window"].fillna(False).astype(bool) & z.notna() & (z <= -threshold)
        conviction = ((-z - threshold) / span).clip(0.0, 1.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        # The identical entry condition becoming false is the only exit.
        df["signal"] = eligible.astype(int)
        df["size"] = (0.20 + 0.75 * conviction).where(eligible, 0.20)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-03-18", periods=4, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = ExpirationDiscountParams()
    derived = pd.DataFrame({"z": [-1.0, -2.0, 0.0, -1.0], "seasonal_window": True}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert np.allclose(result["size"], [0.20, 0.525, 0.95, 0.20])
    cold = strategy.indicators(data, params)
    assert strategy.generate_signals(data, cold, None, params).data["signal"].eq(0).all()
