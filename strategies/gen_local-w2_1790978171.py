from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStaircaseParams:
    streak_bars: int = 2
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[GapStaircaseParams]):
    strategy_id = "gen_local-w2_1790978171"

    @classmethod
    def params_type(cls) -> type[GapStaircaseParams]:
        return GapStaircaseParams

    @staticmethod
    def warmup_bars(params: GapStaircaseParams) -> int:
        if not 1 <= params.streak_bars <= 8:
            raise ValueError("streak_bars must be between 1 and 8")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        return int(params.streak_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GapStaircaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        previous_close = data["close"].shift(1)
        gap = (data["open"] / previous_close.where(previous_close > 0) - 1).replace([np.inf, -np.inf], np.nan)
        previous_gap = gap.shift(1)
        protected = (
            (gap > 0)
            & (data["low"] > previous_close)
            & (data["close"] >= data["open"])
        ).fillna(False)
        shrinking = ((previous_gap > 0) & (gap < previous_gap)).fillna(False)
        eligible = protected & shrinking
        cumulative = eligible.astype(int).cumsum()
        reset = cumulative.where(~eligible).ffill().fillna(0)
        streak = (cumulative - reset).clip(upper=params.streak_bars + 1).astype(int)
        return pd.DataFrame(
            {
                "previous_close": previous_close,
                "gap": gap,
                "previous_gap": previous_gap,
                "protected": protected,
                "shrinking": shrinking,
                "eligible": eligible,
                "streak": streak,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapStaircaseParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["streak"].eq(params.streak_bars).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                exit_at = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rows = [
        [100.0, 101.0, 99.0, 100.0, 1000.0],
        [102.0, 104.0, 101.0, 103.0, 1000.0],
        [104.03, 105.0, 103.5, 104.5, 1000.0],
        [105.0225, 106.0, 104.7, 105.4, 1000.0],
    ] + [[105.4, 106.0, 105.0, 105.4, 1000.0]] * 9
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.date_range("2020-01-01", periods=len(rows)))
    strategy = GeneratedStrategy()
    params = GapStaircaseParams()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert derived["streak"].tolist()[:4] == [0, 0, 1, 2]
    assert result["signal"].tolist() == [0] * 4 + [1] * 7 + [0] * 2
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10
