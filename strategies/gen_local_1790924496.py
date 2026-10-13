from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    peak_lookback: int = 20
    plateau_bars: int = 3
    min_depth: float = 0.012
    max_depth: float = 0.06
    max_depth_spread: float = 0.004
    min_close_location: float = 0.7
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local_1790924496"

    @classmethod
    def params_type(cls) -> type[PlateauParams]:
        return PlateauParams

    def warmup_bars(self, params: PlateauParams) -> int:
        return max(200, max(1, params.peak_lookback) + max(1, params.plateau_bars) - 1)

    def indicators(self, data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        peak_window = max(1, params.peak_lookback)
        plateau_window = max(1, params.plateau_bars)
        result["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        result["peak"] = data["close"].rolling(peak_window, min_periods=peak_window).max()
        result["depth"] = 1.0 - data["close"].div(result["peak"].where(result["peak"] > 0))
        result["depth_min"] = result["depth"].rolling(plateau_window, min_periods=plateau_window).min()
        result["depth_max"] = result["depth"].rolling(plateau_window, min_periods=plateau_window).max()
        result["depth_spread"] = result["depth_max"] - result["depth_min"]
        result["bar_range"] = data["high"] - data["low"]
        result["close_location"] = (data["close"] - data["low"]).div(result["bar_range"].where(result["bar_range"] > 0))
        result["entry"] = (
            (data["close"] > result["ma200"])
            & (result["depth_min"] >= params.min_depth)
            & (result["depth_max"] <= params.max_depth)
            & (result["depth_spread"] <= params.max_depth_spread)
            & (result["close_location"] >= params.min_close_location)
        ).fillna(False)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PlateauParams) -> SignalFrame:
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        entry_bar = 0
        hold_limit = min(2, max(1, params.max_hold_bars))
        target = max(0.0, params.profit_target)

        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    continue

            if active:
                held_bars = i - entry_bar + 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + target)
                if target_hit or held_bars >= hold_limit:
                    active = False
                    # The zero decision exits at the next bar's open.
                    continue
                decisions[i] = 1
            elif entry[i]:
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = PlateauParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (targeted["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
