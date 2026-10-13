from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_min: int = 3
    profit_target: float = 0.04


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791330419"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(65, params.streak_min + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_min < 2 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Require streak_min >= 2 and a finite positive profit_target.")
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff()
        out["direction"] = np.sign(out["change"])
        down = out["direction"].eq(-1)
        out["down_streak"] = down.groupby((~down).cumsum()).cumsum().astype(int)
        out["lagged_direction"] = out["direction"].shift(1)
        out["autocorrelation"] = out["direction"].rolling(63, min_periods=63).corr(out["lagged_direction"])
        out["entry"] = (
            out["down_streak"].shift(1).ge(params.streak_min)
            & out["direction"].eq(1)
            & out["autocorrelation"].lt(-0.10)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if entry_bar < 0 and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar, entry_price = i, opens[i]
            if entry_bar >= 0:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Eighteen held sessions; exit decision fills next open.
                if target_hit or i - entry_bar + 1 >= 18:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({"entry": [True] + [False] * 23}, index=index)
    params = StreakParams()
    timed = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert timed["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    data.loc[index[3], "close"] = 105.0
    targeted = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 1, 1] + [0] * 20
    assert targeted["size"].gt(0).all()
    warm = GeneratedStrategy.indicators(None, data, params)
    assert not warm["entry"].any()


if __name__ == "__main__":
    _self_check()
