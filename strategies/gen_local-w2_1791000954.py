from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    depth_threshold: float = 0.04
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791000954"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 126

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak_21"] = close.rolling(21, min_periods=21).max()
        out["peak_126"] = close.rolling(126, min_periods=126).max()
        out["depth_21"] = (1.0 - close / out["peak_21"]).clip(lower=0.0)
        out["depth_126"] = (1.0 - close / out["peak_126"]).clip(lower=0.0)
        out["previous_depth_21"] = out["depth_21"].shift(1)
        out["entry_ready"] = (
            out["depth_21"].eq(0.0)
            & out["previous_depth_21"].gt(0.0)
            & out["depth_126"].ge(params.depth_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # The prior close's order fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = (
                    valid_entry
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if not valid_entry or target_hit or i - entry_bar + 1 >= 20:
                    active = False
            elif ready[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=24)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entry = pd.DataFrame({"entry_ready": [True] + [False] * 23}, index=index)
    context = StrategyContext("SPY", "1d", 126)
    timed = strategy.generate_signals(bars, entry, context, params).data["signal"]
    assert timed.iloc[0] == 0 and timed.iloc[1:21].eq(1).all()
    assert timed.iloc[21:].eq(0).all()
    bars.iloc[3, bars.columns.get_loc("close")] = 105.0
    targeted = strategy.generate_signals(bars, entry, context, params).data["signal"]
    assert targeted.iloc[1:4].eq(1).all() and targeted.iloc[4:].eq(0).all()
