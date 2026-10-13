from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeasonParams:
    lookback: int = 252
    gap_quantile: float = 0.85
    profit_target: float = 0.025
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapSeasonParams]):
    strategy_id = "gen_local_1791210074"

    @classmethod
    def params_type(cls):
        return GapSeasonParams

    def warmup_bars(self, params: GapSeasonParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapSeasonParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0 < params.gap_quantile < 1:
            raise ValueError("Invalid gap lookback or quantile")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Invalid profit target or holding period")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        opening = data["open"].where(np.isfinite(data["open"]) & (data["open"] > 0))
        out["prior_close"] = close.shift(1)
        out["gap"] = opening / out["prior_close"] - 1.0
        out["gap_threshold"] = out["gap"].abs().shift(1).rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.gap_quantile)
        out["intraday_return"] = close / opening - 1.0
        out["seasonal_window"] = ((data.index.month - 1) % 3 == 0) & (data.index.day <= 14)
        extreme = out["gap"].abs() > out["gap_threshold"]
        long_entry = (out["gap"] < 0) & (out["intraday_return"] > 0) & (close < out["prior_close"])
        short_entry = (out["gap"] > 0) & (out["intraday_return"] < 0) & (close > out["prior_close"])
        eligible = extreme & out["seasonal_window"]
        out["entry"] = np.select([eligible & long_entry, eligible & short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapSeasonParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    continue
                age = i - entry_bar + 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or age >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-02", periods=13)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    data.loc[index[2], "close"] = 103.0
    data.loc[index[6], "close"] = 97.0
    entries = pd.DataFrame({"entry": [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, entries, None, GapSeasonParams(hold_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
