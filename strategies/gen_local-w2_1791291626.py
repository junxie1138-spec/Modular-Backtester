from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    change_threshold: float = 1.5
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local-w2_1791291626"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError("change_threshold must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(5, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["recent"] = out["acceleration"].rolling(4, min_periods=4).mean()
        out["baseline"] = out["acceleration"].rolling(12, min_periods=12).mean().shift(4)
        out["baseline_std"] = out["acceleration"].rolling(12, min_periods=12).std().shift(4)
        out["change_score"] = (
            (out["recent"] - out["baseline"])
            / (out["baseline_std"].where(out["baseline_std"] > 0) * np.sqrt(1.0 / 4 + 1.0 / 12))
        ).replace([np.inf, -np.inf], np.nan)
        out["regime"] = (
            (out["change_score"] > params.change_threshold)
            & (out["baseline"] < 0)
            & (out["recent"] > 0)
            & (out["roc"] > 0)
        ).fillna(False)
        out["entry"] = out["regime"] & ~out["regime"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if entry_bar == i:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                # Entry is the current open after yesterday's decision.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= 18
                if target_hit or time_up:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_price = np.nan
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AccelerationParams()
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.iloc[2, 0] = True
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed["signal"].iloc[:3].eq(0).all()
    assert timed["signal"].iloc[3:21].eq(1).all()
    assert timed["signal"].iloc[21:].eq(0).all()
    data.iloc[7, data.columns.get_loc("close")] = 105.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted["signal"].iloc[3:8].eq(1).all()
    assert targeted["signal"].iloc[8:].eq(0).all()
    derived = strategy.indicators(data, params)
    assert not derived["entry"].iloc[:21].any()
    assert derived.index.equals(data.index)
    prefix = strategy.generate_signals(data.iloc[:8], indicators.iloc[:8], None, params).data
    assert prefix.equals(targeted.iloc[:8])


if __name__ == "__main__":
    _self_check()
