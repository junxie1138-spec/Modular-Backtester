from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    anchor_days: int = 12
    pullback_days: int = 5
    atr_days: int = 14
    min_depth_atr: float = 1.5
    min_gap_atr: float = 0.25
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1791304598"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return int(max(200, params.anchor_days + 1,
                       params.pullback_days + 1, params.atr_days + 1))

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = previous
        out["gap"] = data["open"] - previous
        out["anchor_high"] = data["high"].shift(1).rolling(params.anchor_days, min_periods=params.anchor_days).max()
        out["anchor_low"] = data["low"].shift(1).rolling(params.anchor_days, min_periods=params.anchor_days).min()
        out["recent_high"] = data["high"].shift(1).rolling(params.pullback_days, min_periods=params.pullback_days).max()
        out["recent_low"] = data["low"].shift(1).rolling(params.pullback_days, min_periods=params.pullback_days).min()
        out["long_midpoint"] = (out["anchor_high"] + out["recent_low"]) / 2.0
        out["short_midpoint"] = (out["anchor_low"] + out["recent_high"]) / 2.0
        out["long_depth"] = out["anchor_high"] - previous
        out["short_depth"] = previous - out["anchor_low"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRepairParams) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & indicators["atr"].gt(0)
        close = data["close"]
        opening = data["open"]
        previous = indicators["previous_close"]
        atr = indicators["atr"]
        long_entry = (
            valid
            & close.gt(indicators["ma200"])
            & indicators["long_depth"].ge(params.min_depth_atr * atr)
            & previous.lt(indicators["long_midpoint"])
            & indicators["gap"].ge(params.min_gap_atr * atr)
            & opening.gt(indicators["long_midpoint"])
            & close.ge(indicators["long_midpoint"])
            & close.lt(opening)
        )
        short_entry = (
            valid
            & close.lt(indicators["ma200"])
            & indicators["short_depth"].ge(params.min_depth_atr * atr)
            & previous.gt(indicators["short_midpoint"])
            & indicators["gap"].le(-params.min_gap_atr * atr)
            & opening.lt(indicators["short_midpoint"])
            & close.le(indicators["short_midpoint"])
            & close.gt(opening)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = opening.to_numpy(dtype=float)
        closes = close.to_numpy(dtype=float)
        atr_values = atr.to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry = 0.0
        entry_bar = 0
        stop_profit = 0.0
        armed = False
        hold_days = max(3, min(5, int(params.hold_days)))

        for i in range(len(data)):
            # A prior close's entry decision fills at this bar's open.
            if side == 0 and i > 0 and decisions[i - 1] != 0:
                if (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr_values[i - 1]) and atr_values[i - 1] > 0):
                    side = int(decisions[i - 1])
                    entry = opens[i]
                    entry_bar = i
                    stop_profit = -params.initial_stop_atr * atr_values[i - 1]
                    armed = False

            if side != 0:
                exit_now = i - entry_bar + 1 >= hold_days
                if np.isfinite(closes[i]):
                    profit = side * (closes[i] - entry)
                    exit_now = exit_now or profit <= stop_profit
                    if not exit_now:
                        if profit >= params.breakeven_pct * entry:
                            armed = True
                        if armed:
                            # Signed profit coordinates make the stop increase for both directions.
                            stop_profit = max(stop_profit, 0.0)
                            if np.isfinite(atr_values[i]) and atr_values[i] > 0:
                                stop_profit = max(stop_profit, profit - params.trail_atr * atr_values[i])
                            exit_now = profit <= stop_profit
                if exit_now:
                    side = 0
                    decisions[i] = 0
                    continue
                decisions[i] = side
            else:
                decisions[i] = int(candidates[i])

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    for direction in (1, -1):
        closes = np.array([100.0, 102.0, 103.0, 101.0, 100.0, 100.0, 100.0])
        if direction == -1:
            closes = 200.0 - closes
        opens = closes.copy()
        opens[0] = 101.0 if direction == 1 else 99.0
        opens[1] = 100.0
        data = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1,
                             "low": np.minimum(opens, closes) - 1,
                             "close": closes, "volume": 1000.0}, index=index)
        values = pd.DataFrame({"atr": 1.0, "ma200": 90.0 if direction == 1 else 110.0,
                               "previous_close": 100.0, "gap": 0.0,
                               "anchor_high": 110.0, "anchor_low": 90.0,
                               "recent_high": 106.0, "recent_low": 94.0,
                               "long_midpoint": 105.0, "short_midpoint": 95.0,
                               "long_depth": 0.0, "short_depth": 0.0}, index=index)
        values.loc[index[0], "previous_close"] = 98.0 if direction == 1 else 102.0
        values.loc[index[0], "gap"] = 3.0 * direction
        if direction == 1:
            values.loc[index[0], "long_midpoint"] = 99.0
            values.loc[index[0], "long_depth"] = 3.0
        else:
            values.loc[index[0], "short_midpoint"] = 101.0
            values.loc[index[0], "short_depth"] = 3.0
        strategy = GeneratedStrategy.__new__(GeneratedStrategy)
        result = strategy.generate_signals(data, values, None, GapRepairParams()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0]
        assert result["size"].gt(0).all()
        values.loc[:, "gap"] = 0.0
        assert strategy.generate_signals(data, values, None, GapRepairParams()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
