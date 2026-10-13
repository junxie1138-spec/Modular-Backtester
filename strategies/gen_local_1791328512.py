from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BrakingParams:
    percentile_window: int = 126
    streak_percentile: float = 0.80
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[BrakingParams]):
    strategy_id = "gen_local_1791328512"

    @classmethod
    def params_type(cls) -> type[BrakingParams]:
        return BrakingParams

    def warmup_bars(self, params: BrakingParams) -> int:
        return int(params.percentile_window + 4)

    def indicators(self, data: pd.DataFrame, params: BrakingParams) -> pd.DataFrame:
        if params.percentile_window < 10:
            raise ValueError("percentile_window must be at least 10")
        if not 0.0 < params.streak_percentile < 1.0:
            raise ValueError("streak_percentile must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")

        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = returns.shift(1)
        braking = ((returns < 0) & (previous < 0) & (returns > previous)).fillna(False)
        groups = (~braking).cumsum()
        streak = braking.astype(int).groupby(groups).cumsum()
        threshold = streak.where(streak > 0).rolling(
            params.percentile_window, min_periods=10
        ).quantile(params.streak_percentile).shift(2)

        out["return"] = returns
        out["braking"] = braking
        out["braking_streak"] = streak
        out["streak_threshold"] = threshold
        # At close N, compare the completed streak at N-1 with history through N-2.
        out["entry"] = (
            (returns > 0)
            & (streak.shift(1) > 0)
            & (streak.shift(1) >= threshold)
            & threshold.notna()
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BrakingParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_hit = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_hit:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i]:
                entry_bar = i + 1
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 107.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    params = BrakingParams(hold_bars=3)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 106.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    data["close"] = [100.0, 95.0, 92.0, 90.5, 91.0, 92.0]
    derived = strategy.indicators(data, params)
    assert derived["braking_streak"].tolist() == [0, 0, 1, 2, 0, 0]
    assert not derived["entry"].any()


if __name__ == "__main__":
    _self_check()
