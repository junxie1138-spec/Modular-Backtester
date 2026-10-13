from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryGapParams:
    min_gap: float = 0.003
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ExpiryGapParams]):
    strategy_id = "gen_local_1791263798"

    @classmethod
    def params_type(cls):
        return ExpiryGapParams

    @staticmethod
    def warmup_bars(params: ExpiryGapParams) -> int:
        return 200

    @staticmethod
    def indicators(data: pd.DataFrame, params: ExpiryGapParams) -> pd.DataFrame:
        if not np.isfinite(params.min_gap) or params.min_gap <= 0:
            raise ValueError("min_gap must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["gap"] = (data["open"] / previous - 1).replace([np.inf, -np.inf], np.nan)
        out["unrepaired_discount"] = (1 - data["close"] / previous).replace([np.inf, -np.inf], np.nan)
        out["expiry_friday"] = (
            (data.index.dayofweek == 4)
            & (data.index.day >= 15)
            & (data.index.day <= 21)
        )
        out["entry"] = (
            out["expiry_friday"]
            & (data["close"] > out["ma200"])
            & (out["gap"] <= -params.min_gap)
            & (out["unrepaired_discount"] >= params.min_gap / 2)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpiryGapParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        entry_price = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # The preceding close's decision fills at today's open.
                entry_price = opens[i]
                held = 0
            if entry_price > 0:
                held += 1
                # Close-observed exits fill at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                if target_hit or held >= params.max_hold_bars:
                    entry_price = 0.0
                    held = 0
                    continue
                decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ExpiryGapParams()
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2024-01-19", periods=4, freq="B"),
    )
    derived = pd.DataFrame({"entry": [True, False, False, False]}, index=bars.index)
    signals = GeneratedStrategy.generate_signals(bars, derived, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0]
    bars.loc[bars.index[1], "close"] = 100.8
    signals = GeneratedStrategy.generate_signals(bars, derived, None, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0]
    assert (signals["size"] > 0).all()
