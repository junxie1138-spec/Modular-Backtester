from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAccelerationParams:
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SeasonalAccelerationParams]):
    strategy_id = "gen_local_1791344471"

    @classmethod
    def params_type(cls) -> type[SeasonalAccelerationParams]:
        return SeasonalAccelerationParams

    def warmup_bars(self, params: SeasonalAccelerationParams) -> int:
        # Two five-bar price intervals, 756 observations, one lag.
        return 767

    def indicators(self, data: pd.DataFrame, params: SeasonalAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.div(close.shift(5)).sub(1)
        out["acceleration"] = out["roc"].diff(5)
        months = data.index.year * 12 + data.index.month
        out["month_session"] = pd.Series(1, index=data.index).groupby(months).cumsum()
        out["opening_window"] = out["month_session"].le(3)
        out["noise"] = out["acceleration"].abs().rolling(63, min_periods=63).mean()
        valid = out["opening_window"] & np.isfinite(out["acceleration"])
        out["seasonal_sample"] = out["acceleration"].where(valid, 0.0)
        out["seasonal_count"] = valid.astype(float).rolling(756, min_periods=756).sum().shift(1)
        out["seasonal_sum"] = out["seasonal_sample"].rolling(756, min_periods=756).sum().shift(1)
        out["seasonal_template"] = out["seasonal_sum"].div(out["seasonal_count"].replace(0, np.nan))
        ready = (
            out["opening_window"]
            & out["seasonal_count"].ge(24)
            & out["noise"].gt(0)
            & out["acceleration"].abs().gt(0.5 * out["noise"])
            & out["seasonal_template"].abs().gt(0.1 * out["noise"])
            & out["acceleration"].mul(out["seasonal_template"]).gt(0)
            & np.isfinite(out["acceleration"])
            & np.isfinite(out["seasonal_template"])
        )
        out["entry_direction"] = np.sign(out["acceleration"]).where(ready, 0).fillna(0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalAccelerationParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if isinstance(params.holding_bars, bool) or not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        months = data.index.year * 12 + data.index.month
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        last_entry_month = -1
        for i in range(len(data)):
            if direction:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        direction = 0
                        continue
                timed_out = i - entry_bar + 1 >= params.holding_bars
                gain = direction * (closes[i] / entry_price - 1.0)
                if timed_out or (np.isfinite(gain) and gain >= params.profit_target):
                    direction = 0
                    continue
            elif (
                entries[i] in (-1, 1)
                and months[i] != last_entry_month
                and np.isfinite(closes[i])
                and closes[i] > 0
            ):
                direction = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                last_entry_month = months[i]
            signals[i] = direction
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-02", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_direction": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    params = SeasonalAccelerationParams(profit_target=0.04, holding_bars=3)
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data.loc[index[1], "close"] = 105.0
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0]
    indicators["entry_direction"] = [-1, 0, 0, 0, 0, 0, 0]
    data.loc[index[1], "close"] = 95.0
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0]
