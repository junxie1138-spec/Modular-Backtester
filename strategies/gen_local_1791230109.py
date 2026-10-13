from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidenceParams:
    window: int = 20
    history: int = 126
    position_quantile: float = 0.65
    streak_quantile: float = 0.85


class GeneratedStrategy(BaseStrategy[ResidenceParams]):
    strategy_id = "gen_local_1791230109"

    @classmethod
    def params_type(cls) -> type[ResidenceParams]:
        return ResidenceParams

    def warmup_bars(self, params: ResidenceParams) -> int:
        return int(params.window + params.history + 1)

    def indicators(self, data: pd.DataFrame, params: ResidenceParams) -> pd.DataFrame:
        if params.window < 2 or params.history < 2:
            raise ValueError("window and history must be at least two")
        if not 0.5 < params.position_quantile < 1.0:
            raise ValueError("position_quantile must lie between 0.5 and 1")
        if not 0.0 < params.streak_quantile < 1.0:
            raise ValueError("streak_quantile must lie between 0 and 1")

        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        prior = close.shift(1).rolling(params.window, min_periods=params.window)
        out["upper_band"] = prior.quantile(params.position_quantile)
        out["lower_band"] = prior.quantile(1.0 - params.position_quantile)
        valid = close.notna() & out["upper_band"].gt(out["lower_band"])
        out["side"] = np.select(
            [valid & close.gt(out["upper_band"]), valid & close.lt(out["lower_band"])],
            [1, -1], default=0,
        ).astype(int)

        for label, direction in (("upper", 1), ("lower", -1)):
            occupied = out["side"].eq(direction)
            out[label + "_streak"] = occupied.astype(int).groupby((~occupied).cumsum()).cumsum()
            past_streaks = out[label + "_streak"].where(out[label + "_streak"].gt(0)).shift(1)
            out[label + "_threshold"] = past_streaks.rolling(
                params.history, min_periods=1
            ).quantile(params.streak_quantile)

        out["ready"] = np.arange(len(data)) >= self.warmup_bars(params) - 1
        long_entry = (
            out["ready"] & out["upper_streak"].gt(0)
            & out["upper_streak"].ge(out["upper_threshold"])
        )
        short_entry = (
            out["ready"] & out["lower_streak"].gt(0)
            & out["lower_streak"].ge(out["lower_threshold"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ResidenceParams,
    ) -> SignalFrame:
        state = 0
        signals = np.zeros(len(data), dtype=int)
        events = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i, event in enumerate(events):
            if state == 0:
                state = int(event)
            elif event == -state:
                state = 0
            signals[i] = state

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ResidenceParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    events = pd.DataFrame({"entry": [1, 0, -1, -1, 0, 1, 1, 0]}, index=index)
    result = strategy.generate_signals(data, events, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert flat["entry"].eq(0).all()
