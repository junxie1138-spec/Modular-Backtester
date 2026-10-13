from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    correlation_threshold: float = 0.30
    innovation_threshold: float = 0.20
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local-w2_1791210954"

    @classmethod
    def params_type(cls):
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        if not (-1.0 <= params.correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be between -1 and 1")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError("innovation_threshold must be finite and nonnegative")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(params.hold_bars, int) or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer between 3 and 5")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where((close > 0) & np.isfinite(close)))
        price = out["log_close"]
        out["return"] = price.diff()
        out["slope"] = (1.5 * price + 0.5 * price.shift(1)
                        - 0.5 * price.shift(2) - 1.5 * price.shift(3)) / 5.0
        out["innovation"] = out["slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["volatility"] = out["return"].rolling(4, min_periods=4).std(ddof=0)
        out["innovation_strength"] = out["innovation"].div(
            out["volatility"].where(out["volatility"] > 0)
        )
        out["innovation_correlation"] = out["innovation"].rolling(
            4, min_periods=4
        ).corr(out["lagged_innovation"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["slope"] > 0)
            & (out["innovation_strength"] > params.innovation_threshold)
            & (out["innovation_correlation"] > params.correlation_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: InnovationParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held_bars = 0

        for i in range(len(data)):
            if active:
                # The preceding decision fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                held_bars += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held_bars >= params.hold_bars:
                    active = False
                    held_bars = 0
                    continue
                decisions[i] = 1
            elif entries[i]:
                active = True
                held_bars = 0
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = InnovationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
