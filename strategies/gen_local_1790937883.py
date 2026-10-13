from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    gap_window: int = 4
    volatility_window: int = 20
    acceptance_window: int = 5
    acceleration_threshold: float = 0.6
    acceptance_threshold: float = 0.4
    minimum_gap: float = 0.001
    retention_fraction: float = 0.5
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local_1790937883"

    @classmethod
    def params_type(cls):
        return GapAccelerationParams

    def warmup_bars(self, params: GapAccelerationParams) -> int:
        return int(max(2 * params.gap_window + 1,
                       params.volatility_window + 1,
                       params.acceptance_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapAccelerationParams) -> pd.DataFrame:
        if min(params.gap_window, params.volatility_window,
               params.acceptance_window, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (params.acceleration_threshold > 0
                and 0 < params.acceptance_threshold <= 1
                and params.minimum_gap >= 0
                and 0 <= params.retention_fraction <= 1
                and params.profit_target > 0):
            raise ValueError("Invalid signal or exit thresholds.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        ratio = data["open"].where(lambda x: x > 0) / previous_close
        out["gap"] = ratio - 1.0
        out["log_gap"] = np.log(ratio)
        out["gap_roc"] = out["log_gap"].rolling(params.gap_window).sum()
        out["previous_gap_roc"] = out["gap_roc"].shift(params.gap_window)
        out["gap_scale"] = out["log_gap"].rolling(params.volatility_window).std(ddof=0)
        scale = out["gap_scale"].where(out["gap_scale"] > 0)
        out["acceleration"] = (
            (out["gap_roc"] - out["previous_gap_roc"])
            / (scale * np.sqrt(2.0 * params.gap_window))
        )
        out["close_from_previous"] = data["close"] / previous_close - 1.0
        direction = np.sign(out["gap"])
        accepted = (
            (out["gap"].abs() >= params.minimum_gap)
            & (direction * out["close_from_previous"]
               >= params.retention_fraction * out["gap"].abs())
        )
        out["retained_gap_vote"] = (direction * accepted.astype(float)).where(out["gap"].notna())
        out["acceptance"] = out["retained_gap_vote"].rolling(params.acceptance_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapAccelerationParams) -> SignalFrame:
        acceleration = indicators["acceleration"]
        acceptance = indicators["acceptance"]
        long_entry = ((acceleration >= params.acceleration_threshold)
                      & (acceptance >= params.acceptance_threshold))
        short_entry = ((acceleration <= -params.acceleration_threshold)
                       & (acceptance <= -params.acceptance_threshold))
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)],
                            [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # An entry chosen at the preceding close fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                held_bars = i - entry_bar + 1
                if gain >= params.profit_target or held_bars >= params.max_hold_bars:
                    position = 0
                    # Exit at the following open; do not re-enter on this decision bar.
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame({"open": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], ["open", "close"]] = [120.0, 124.0]
    data.loc[index[5], "close"] = 97.0
    data["high"] = data[["open", "close"]].max(axis=1) + 1.0
    data["low"] = data[["open", "close"]].min(axis=1) - 1.0
    votes = pd.DataFrame({"acceleration": 0.0, "acceptance": 0.0}, index=index)
    votes.loc[index[0]] = [1.0, -1.0]
    votes.loc[index[1]] = [1.0, 1.0]
    votes.loc[index[4]] = [-1.0, -1.0]
    votes.loc[index[7]] = [1.0, 1.0]
    strategy = GeneratedStrategy()
    params = GapAccelerationParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, votes, ctx, params).data
    expected = np.zeros(20, dtype=int)
    expected[2] = 1
    expected[5] = -1
    expected[8:16] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(data.index) and (result["size"] > 0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    derived = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
