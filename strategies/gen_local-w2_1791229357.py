from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    atr_window: int = 14
    gap_atr: float = 0.9
    range_atr: float = 0.6
    close_location: float = 0.7
    profit_target: float = 0.01
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local-w2_1791229357"

    @classmethod
    def params_type(cls) -> type[GapCapacityParams]:
        return GapCapacityParams

    @staticmethod
    def warmup_bars(params: GapCapacityParams) -> int:
        return int(max(200, params.atr_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if params.atr_window < 1 or params.gap_atr <= 0 or params.range_atr <= 0:
            raise ValueError("ATR window and volatility thresholds must be positive")
        if not 0 <= params.close_location <= 1:
            raise ValueError("close_location must be between zero and one")
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        result = pd.DataFrame(index=data.index)
        result["true_range"] = true_range
        result["prior_atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().shift(1)
        result["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        denominator = result["prior_atr"].where(result["prior_atr"] > 0)
        result["gap_atr"] = (data["open"] - previous_close) / denominator
        daily_range = data["high"] - data["low"]
        result["range_atr"] = daily_range / denominator
        result["close_location"] = (data["close"] - data["low"]) / daily_range.where(daily_range > 0)
        result = result.replace([np.inf, -np.inf], np.nan)
        result["entry"] = (
            (data["close"] > result["ma200"])
            & (result["gap_atr"] >= params.gap_atr)
            & (result["range_atr"] <= params.range_atr)
            & (result["close_location"] >= params.close_location)
            & (data["close"] >= data["open"])
        ).fillna(False)
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The previous close's decision fills at this session's open.
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or target_hit or time_expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    decisions[i] = 1
            elif entries[i]:
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": [100.0, 101.2, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False]}, index=index)
    params = GapCapacityParams()
    target = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert target["signal"].tolist() == [0, 1, 0, 0]
    data["close"] = 100.0
    timed = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0]
    assert timed["size"].gt(0).all()
