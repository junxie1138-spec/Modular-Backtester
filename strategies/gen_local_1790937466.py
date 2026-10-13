from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    repair_streak: int = 3
    min_drawdown: float = 0.025
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790937466"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not 1 <= params.repair_streak <= 8:
            raise ValueError("repair_streak must be between 1 and 8")
        if not np.isfinite(params.min_drawdown) or not 0 < params.min_drawdown < 1:
            raise ValueError("min_drawdown must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        if params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive")

        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(9, min_periods=9).max()
        out["drawdown"] = (1.0 - data["close"] / out["peak"].where(out["peak"] > 0)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out["rising_low"] = data["low"].diff().gt(0).fillna(False)
        consecutive = pd.Series(True, index=data.index)
        streak = pd.Series(0, index=data.index, dtype=int)
        for lag in range(8):
            consecutive = consecutive & out["rising_low"].shift(lag, fill_value=False)
            streak = streak + consecutive.astype(int)
        out["repair_streak"] = streak
        out["bullish_close"] = data["close"].gt(data["open"]).fillna(False)
        out["entry"] = (
            out["peak"].notna()
            & out["repair_streak"].eq(params.repair_streak)
            & out["drawdown"].ge(params.min_drawdown)
            & out["bullish_close"]
        ).fillna(False)
        out["peak"] = out["peak"].fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # Read the next-bar opening reference only when that bar arrives.
                if i == entry_bar:
                    entry_price = opens[i]
                held_bars = i - entry_bar + 1
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = int(active)

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    close = np.r_[np.full(9, 110.0), np.full(31, 100.0)]
    opening = np.r_[np.full(9, 110.0), np.full(31, 99.5)]
    low = np.r_[np.full(9, 109.0), [95.0, 96.0, 97.0, 98.0], np.full(27, 99.0)]
    data = pd.DataFrame(
        {"open": opening, "high": close + 1.0, "low": low, "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert signals["signal"].iloc[12] == 0 and signals["signal"].iloc[13] == 1
    assert signals["signal"].iloc[30] == 1 and signals["signal"].iloc[31] == 0
    data.loc[index[14], ["close", "high"]] = [104.0, 105.0]
    signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert signals["signal"].iloc[14] == 1 and signals["signal"].iloc[15] == 0
