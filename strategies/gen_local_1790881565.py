from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryGapParams:
    gap_min: float = 0.0015
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ExpiryGapParams]):
    strategy_id = "gen_local_1790881565"

    @classmethod
    def params_type(cls):
        return ExpiryGapParams

    def warmup_bars(self, params: ExpiryGapParams) -> int:
        return 2

    def indicators(self, data: pd.DataFrame, params: ExpiryGapParams) -> pd.DataFrame:
        if not (0 < params.gap_min < 1 and 0 < params.profit_target < 1
                and params.max_hold_bars in (1, 2)):
            raise ValueError("Positive fractional thresholds and a one- or two-bar holding limit required.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).fillna(0.0)
        out["gap"] = (data["open"] / out["prior_close"].where(
            out["prior_close"] > 0
        ) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expiry_monday"] = (
            (data.index.dayofweek == 0)
            & (data.index.day >= 18) & (data.index.day <= 24)
        )
        out["valid"] = (
            np.isfinite(data["open"]) & np.isfinite(data["close"])
            & (data["open"] > 0) & (data["close"] > 0)
            & (out["prior_close"] > 0)
        )
        out["entry"] = (
            out["valid"] & out["expiry_monday"]
            & (out["gap"] <= -params.gap_min)
            & (data["close"] < out["prior_close"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ExpiryGapParams,
    ) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's decision enters at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                # Closing target confirmation exits at the following open.
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= params.max_hold_bars:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i] and i + 1 < len(data):
                entry_bar = i + 1
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    dates = pd.to_datetime(["2023-03-17", "2023-03-20", "2023-03-21",
                            "2023-03-22", "2023-03-23"])
    data = pd.DataFrame({
        "open": [100.0, 99.0, 99.0, 99.0, 99.0],
        "high": [101.0, 100.0, 101.0, 100.0, 100.0],
        "low": [99.0, 98.0, 98.0, 98.0, 98.0],
        "close": [100.0, 99.0, 100.0, 99.0, 99.0],
        "volume": [1000.0] * 5,
    }, index=dates)
    strategy = GeneratedStrategy()
    params = ExpiryGapParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=2)
    target = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert target["signal"].tolist() == [0, 0, 1, 0, 0]
    data.loc[dates[2], "close"] = 99.0
    timed = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (timed["size"] > 0).all() and strategy.warmup_bars(params) <= 10
    data.index = data.index + pd.Timedelta(days=7)
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
