from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_bars: int = 3
    max_upper_wick: float = 0.35
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790981261"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 1 <= params.streak_bars <= 8:
            raise ValueError("streak_bars must be between 1 and 8")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        if not 0.0 <= params.max_upper_wick <= 1.0:
            raise ValueError("max_upper_wick must be between 0 and 1")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        return int(params.streak_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["body_top"] = data[["open", "close"]].max(axis=1)
        out["upper_wick"] = (
            (data["high"] - out["body_top"]) / out["range"]
        ).replace([np.inf, -np.inf], np.nan)
        out["low_lift"] = data["low"] > data["low"].shift(1)
        out["rejection_fades"] = (
            out["upper_wick"] < out["upper_wick"].shift(1)
        )
        out["step"] = out["low_lift"] & out["rejection_fades"]
        consecutive = pd.Series(True, index=data.index)
        out["streak"] = 0
        for offset in range(params.streak_bars):
            consecutive &= out["step"].shift(offset, fill_value=False)
            out["streak"] += consecutive.astype(int)
        out["close_rises"] = data["close"] > data["close"].shift(1)
        out["setup"] = (
            (out["streak"] >= params.streak_bars)
            & (out["upper_wick"] <= params.max_upper_wick)
            & out["close_rises"]
        ).fillna(False)
        out["entry"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        decision = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if entry_bar < 0 and i > 0 and decision[i - 1] == 1:
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                age = i - entry_bar + 1
                # Close-confirmed target; exit fills at the following open.
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.max_hold_bars:
                    entry_bar = -1
                else:
                    decision[i] = 1
            elif i >= warmup - 1 and entries[i]:
                decision[i] = 1
        df = pd.DataFrame({"signal": decision, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    close = np.arange(100.0, 108.0)
    data = pd.DataFrame({
        "open": close - 0.5,
        "high": close + np.array([3, 2.5, 2, 1.5, 1, 0.8, 0.6, 0.4]),
        "low": close - 2,
        "close": close,
        "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=8))
    strategy = GeneratedStrategy()
    for target, expected in (
        (1.0, [0, 0, 0, 0, 0, 1, 1, 0]),
        (0.004, [0, 0, 0, 0, 0, 1, 0, 0]),
    ):
        params = Params(profit_target=target)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(
            data, strategy.indicators(data, params), ctx, params
        ).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
