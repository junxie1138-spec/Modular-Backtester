from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    baseline_bars: int = 20
    min_streak: int = 3
    compression_ratio: float = 0.65
    close_location_min: float = 0.65
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791290926"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(200, params.baseline_bars + 2, params.min_streak + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        daily_range = data["high"] - data["low"]
        out["normalized_range"] = daily_range.where(daily_range > 0).div(
            previous_close.where(previous_close > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["range_baseline"] = out["normalized_range"].rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).median().shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["close_location"] = (data["close"] - data["low"]).div(
            daily_range.where(daily_range > 0)
        ).replace([np.inf, -np.inf], np.nan)
        shrinking = (
            out["normalized_range"].lt(out["normalized_range"].shift(1))
            & out["normalized_range"].gt(0)
            & out["range_baseline"].notna()
        ).fillna(False)
        out["compression_streak"] = shrinking.groupby((~shrinking).cumsum()).cumsum().astype(int)
        out["entry"] = (
            out["compression_streak"].eq(params.min_streak)
            & out["normalized_range"].le(params.compression_ratio * out["range_baseline"])
            & out["close_location"].ge(params.close_location_min)
            & data["close"].gt(out["ma200"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        previous = 0
        for i in range(len(data)):
            # The previous close's entry decision fills at this session's open.
            if previous == 1 and entry_bar < 0:
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                # Close-confirmed exits execute at the following session's open.
                if (
                    not valid
                    or closes[i] >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
            previous = decisions[i]
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=4)
    entries = pd.DataFrame({"entry": [True, False, False, False]}, index=index)
    for closes, expected in (
        ([100.0, 100.0, 100.0, 100.0], [0, 1, 1, 0]),
        ([100.0, 101.0, 100.0, 100.0], [0, 1, 0, 0]),
    ):
        data = pd.DataFrame(
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": closes, "volume": 1000.0},
            index=index,
        )
        result = GeneratedStrategy.generate_signals(data, entries, None, params)
        assert result.data["signal"].tolist() == expected
