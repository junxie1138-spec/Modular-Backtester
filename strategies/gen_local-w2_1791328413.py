from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_cap: int = 20
    percentile_window: int = 126
    streak_percentile: float = 0.85
    drift_bars: int = 20
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791328413"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(params.streak_cap + params.percentile_window + 2,
                       params.drift_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if min(params.streak_cap, params.percentile_window,
               params.drift_bars, params.hold_bars) < 1:
            raise ValueError("Bar counts must be positive")
        if not 0.0 < params.streak_percentile < 1.0:
            raise ValueError("streak_percentile must be between zero and one")

        out = pd.DataFrame(index=data.index)
        out["higher_low"] = data["low"].diff().gt(0)
        groups = (~out["higher_low"]).cumsum()
        out["streak"] = out["higher_low"].astype(int).groupby(groups).cumsum().clip(
            upper=params.streak_cap
        )
        # Bound the streak's effective memory before building its reference distribution.
        out.loc[out.index[:params.streak_cap + 1], "streak"] = np.nan
        out["streak_threshold"] = out["streak"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.streak_percentile)
        out["drift"] = data["close"].diff(params.drift_bars)
        out["exceptional"] = (
            out["streak_threshold"].notna()
            & out["streak"].gt(out["streak_threshold"])
        ).fillna(False)
        out["entry"] = (
            out["exceptional"]
            & ~out["exceptional"].shift(1, fill_value=False)
            & out["drift"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Emit flat on expiry even if another entry appears on this bar.
                    exit_bar = -1
            elif entries[i]:
                exit_bar = i + params.hold_bars
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = StreakParams(hold_bars=8)
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.iloc[[0, 3, 8, 10], 0] = True
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 2 + [1] * 8 + [0]
    assert result["size"].gt(0).all()
