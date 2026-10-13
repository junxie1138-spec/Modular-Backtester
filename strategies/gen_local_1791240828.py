from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParticipationParams:
    gap_window: int = 8
    volume_window: int = 10
    gap_breadth_min: float = 0.25
    volume_balance_min: float = 0.10


class GeneratedStrategy(BaseStrategy[GapParticipationParams]):
    strategy_id = "gen_local_1791240828"

    @classmethod
    def params_type(cls) -> type[GapParticipationParams]:
        return GapParticipationParams

    def warmup_bars(self, params: GapParticipationParams) -> int:
        return int(max(params.gap_window + 1, params.volume_window))

    def indicators(self, data: pd.DataFrame, params: GapParticipationParams) -> pd.DataFrame:
        if params.gap_window < 1 or params.volume_window < 1:
            raise ValueError("Lookback windows must be positive")
        if not (0.0 <= params.gap_breadth_min <= 1.0 and 0.0 <= params.volume_balance_min <= 1.0):
            raise ValueError("Thresholds must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / out["prior_close"] - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_vote"] = np.sign(out["gap"])
        out["gap_breadth"] = out["gap_vote"].rolling(params.gap_window, min_periods=params.gap_window).mean()
        out["session_vote"] = np.sign(data["close"] - data["open"])
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0).replace([np.inf, -np.inf], np.nan)
        out["signed_volume"] = out["valid_volume"] * out["session_vote"]
        out["volume_total"] = out["valid_volume"].rolling(params.volume_window, min_periods=params.volume_window).sum()
        out["signed_volume_total"] = out["signed_volume"].rolling(params.volume_window, min_periods=params.volume_window).sum()
        out["volume_balance"] = out["signed_volume_total"] / out["volume_total"].where(out["volume_total"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapParticipationParams) -> SignalFrame:
        gap_agrees = indicators["gap_breadth"].gt(params.gap_breadth_min)
        volume_agrees = indicators["volume_balance"].gt(params.volume_balance_min)
        eligible = gap_agrees & volume_agrees
        df = pd.DataFrame(index=data.index)
        # The identical conjunction governs entry and continued holding.
        # Its reversal is the only exit; there is no timed exit or stop.
        df["signal"] = eligible.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=20)
    opening = 100.0 + 2.0 * np.arange(20)
    sample = pd.DataFrame({"open": opening, "high": opening + 2.0, "low": opening - 1.0, "close": opening + 1.0, "volume": 100.0}, index=index)
    strategy = GeneratedStrategy()
    params = GapParticipationParams()
    derived = strategy.indicators(sample, params)
    derived.loc[index[12], "volume_balance"] = -1.0
    derived.loc[index[14], "gap_breadth"] = -1.0
    result = strategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].iloc[:10].eq(0).all()
    assert result["signal"].iloc[10] == 1
    assert result["signal"].iloc[12:16].tolist() == [1, 0, 1, 0]
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
