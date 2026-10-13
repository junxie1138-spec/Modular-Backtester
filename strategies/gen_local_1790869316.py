from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790869316"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 44

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        result = pd.DataFrame(index=data.index)
        result["return"] = close.pct_change(fill_method=None)
        result["lagged_return"] = result["return"].shift(1)
        result["roc"] = close.pct_change(3, fill_method=None)
        result["acceleration"] = result["roc"].diff()
        result["acceleration_scale"] = result["acceleration"].rolling(40, min_periods=40).std()
        result["normalized_acceleration"] = result["acceleration"].div(result["acceleration_scale"].replace(0, np.nan))
        result["autocorrelation"] = result["return"].rolling(40, min_periods=40).corr(result["lagged_return"])
        result = result.replace([np.inf, -np.inf], np.nan)
        ready = result.notna().all(axis=1)
        rising = (result["roc"] > 0) & (result["normalized_acceleration"] > 0.8)
        falling = (result["roc"] < 0) & (result["normalized_acceleration"] < -0.8)
        persistent = result["autocorrelation"] > 0.1
        alternating = result["autocorrelation"] < -0.1
        result["entry"] = np.select(
            [ready & ((persistent & rising) | (alternating & falling)),
             ready & ((persistent & falling) | (alternating & rising))],
            [1, -1], default=0,
        ).astype(int)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold not in (3, 4, 5):
            raise ValueError("max_hold must be 3, 4, or 5")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                if entry_bar == -1:
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": 1}, index=index)
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].tolist()
    assert signals == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    data.loc[index[1], "close"] = 102.0
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].tolist()
    assert signals[:3] == [0, 1, 0]
