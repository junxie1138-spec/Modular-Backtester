from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReopeningParams:
    atr_period: int = 20
    range_multiple: float = 1.5
    reversal_atr: float = 0.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ReopeningParams]):
    strategy_id = "gen_local-w2_1791262101"

    @classmethod
    def params_type(cls) -> type[ReopeningParams]:
        return ReopeningParams

    @staticmethod
    def warmup_bars(params: ReopeningParams) -> int:
        if params.atr_period < 2:
            raise ValueError("atr_period must be at least 2")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between 3 and 5")
        if not np.isfinite(params.range_multiple) or params.range_multiple <= 0:
            raise ValueError("range_multiple must be finite and positive")
        if not np.isfinite(params.reversal_atr) or params.reversal_atr <= 0:
            raise ValueError("reversal_atr must be finite and positive")
        return int(max(200, params.atr_period + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReopeningParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        result["previous_close"] = data["close"].shift(1)
        components = pd.concat([
            data["high"] - data["low"],
            (data["high"] - result["previous_close"]).abs(),
            (data["low"] - result["previous_close"]).abs(),
        ], axis=1)
        result["true_range"] = components.max(axis=1, skipna=False)
        result["prior_atr"] = result["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean().shift(1)
        result["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        result["closure_days"] = data.index.to_series().diff().dt.total_seconds() / 86400.0
        denominator = result["prior_atr"].where(result["prior_atr"] > 0)
        result["range_ratio"] = result["true_range"] / denominator
        result["move_atr"] = (data["close"] - result["previous_close"]) / denominator
        result["valid"] = (
            np.isfinite(result[["ma200", "range_ratio", "move_atr"]]).all(axis=1)
            & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & (data["close"] > 0)
        )
        result["reopening_shock"] = (
            result["valid"]
            & result["closure_days"].ge(3)
            & result["range_ratio"].ge(params.range_multiple)
        ).fillna(False)
        long_entry = (
            result["reopening_shock"]
            & data["close"].gt(result["ma200"])
            & result["move_atr"].le(-params.reversal_atr)
        )
        short_entry = (
            result["reopening_shock"]
            & data["close"].lt(result["ma200"])
            & result["move_atr"].ge(params.reversal_atr)
        )
        result["entry_direction"] = np.select(
            [long_entry, short_entry], [1, -1], default=0
        ).astype(int)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReopeningParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        exit_bar = -1
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        for i, entry in enumerate(entries):
            if direction != 0:
                if i == exit_bar:
                    direction = 0
                    continue
            elif entry != 0:
                direction = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({"entry_direction": [1, 0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, ReopeningParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert result.data.index.equals(index)
    assert result.data["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(ReopeningParams(atr_period=220)) == 221
