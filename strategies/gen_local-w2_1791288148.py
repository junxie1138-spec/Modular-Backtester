from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCancellationParams:
    gap_floor: float = 0.002
    cancellation_tolerance: float = 0.35
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GapCancellationParams]):
    strategy_id = "gen_local-w2_1791288148"

    @classmethod
    def params_type(cls):
        return GapCancellationParams

    def warmup_bars(self, params: GapCancellationParams) -> int:
        return 3

    def indicators(self, data: pd.DataFrame, params: GapCancellationParams) -> pd.DataFrame:
        if not (
            np.isfinite(params.gap_floor) and params.gap_floor > 0
            and np.isfinite(params.cancellation_tolerance)
            and 0 < params.cancellation_tolerance < 1
            and np.isfinite(params.profit_target) and params.profit_target > 0
            and isinstance(params.max_hold, int) and 3 <= params.max_hold <= 5
        ):
            raise ValueError("Require positive gap and target, tolerance in (0, 1), and max_hold in [3, 5].")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = data["open"] / previous_close - 1.0
        out["previous_gap"] = out["gap"].shift(1)
        out["cancellation"] = (
            (out["gap"] + out["previous_gap"]).abs()
            / out["previous_gap"].abs().replace(0.0, np.nan)
        )
        out["volume_ratio"] = (
            data["volume"] / data["volume"].shift(1).replace(0.0, np.nan)
        )
        out["repaired"] = data["close"].ge(previous_close)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["gap"].le(-params.gap_floor)
            & out["previous_gap"].ge(params.gap_floor)
            & out["cancellation"].le(params.cancellation_tolerance)
            & out["volume_ratio"].lt(1.0)
            & out["repaired"]
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCancellationParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Read the execution open only after the entry decision.
                if i == entry_bar:
                    entry_price = opens[i]
                profit_exit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_exit = i - entry_bar + 1 >= params.max_hold
                if profit_exit or time_exit:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                raw[i] = 1
            elif entries[i]:
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapCancellationParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] + [110.0] * 7,
        "high": [100.0] + [113.0] * 7,
        "low": [99.0] + [109.0] * 7,
        "close": [100.0] + [110.0] * 7,
        "volume": [1000.0] * 8,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=3)
    entries = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 112.0
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    data = pd.DataFrame({
        "open": [100.0, 101.0, 100.0],
        "high": [100.5, 102.0, 102.0],
        "low": [99.5, 100.5, 99.5],
        "close": [100.0, 101.0, 101.2],
        "volume": [1000.0, 2000.0, 1500.0],
    }, index=index[:3])
    assert strategy.indicators(data, params)["entry"].tolist() == [False, False, True]


if __name__ == "__main__":
    _self_check()
