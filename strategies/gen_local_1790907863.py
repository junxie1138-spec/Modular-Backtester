from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 60
    deep_drawdown: float = 0.04
    shallow_drawdown: float = 0.01
    min_gap: float = 0.003
    profit_target: float = 0.025
    max_hold: int = 8
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790907863"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.peak_window + 1, 3)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (
            params.peak_window >= 2
            and 0 <= params.shallow_drawdown < params.deep_drawdown < 1
            and params.min_gap > 0
            and params.profit_target > 0
            and params.max_hold >= 1
            and 0 < params.size <= 1
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        out["drawdown"] = 1.0 - close.shift(1) / out["peak"]
        out["gap"] = data["open"].where(data["open"] > 0) / close.shift(1) - 1.0
        out["prior_gap"] = out["gap"].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = np.select(
            [
                (out["drawdown"] >= params.deep_drawdown)
                & (out["gap"] >= params.min_gap)
                & (out["prior_gap"] <= -params.min_gap),
                (out["drawdown"] <= params.shallow_drawdown)
                & (out["gap"] <= -params.min_gap)
                & (out["prior_gap"] >= params.min_gap),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                held += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    held = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                held = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(peak_window=3, profit_target=0.025, max_hold=2)
    index = pd.date_range("2020-01-01", periods=8)
    close = np.array([100, 100, 94, 94, 97, 97, 97, 97], dtype=float)
    opening = np.array([100, 100, 98, 95, 94, 97, 97, 97], dtype=float)
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, close) + 1,
         "low": np.minimum(opening, close) - 1, "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    indicators = strategy.indicators(data, params)
    assert indicators["entry"].iloc[3] == 1
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    params.profit_target = 0.5
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    changed = data.copy()
    changed.iloc[2, changed.columns.get_loc("open")] = 101.0
    assert strategy.indicators(changed, params)["entry"].iloc[3] == 0
    short_data = data.copy()
    short_data["close"] = 100.0
    short_data["open"] = [100, 100, 101, 99, 100, 100, 100, 100]
    short_indicators = strategy.indicators(short_data, params)
    assert short_indicators["entry"].iloc[3] == -1
    short_signals = strategy.generate_signals(short_data, short_indicators, ctx, params).data
    assert short_signals["signal"].tolist() == [0, 0, 0, 0, -1, -1, 0, 0]
    assert strategy.warmup_bars(params) >= params.peak_window + 1


if __name__ == "__main__":
    demo()
