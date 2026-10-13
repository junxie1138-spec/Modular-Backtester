from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 3
    autocorr_window: int = 40
    min_autocorr: float = 0.05
    profit_target: float = 0.012
    max_hold_bars: int = 4

    def __post_init__(self):
        if self.streak_length < 2 or self.autocorr_window < 3:
            raise ValueError("Streak and correlation windows are too short")
        if not (-1.0 <= self.min_autocorr <= 1.0):
            raise ValueError("Invalid correlation threshold")
        if not (0.0 < self.profit_target < 1.0) or not (3 <= self.max_hold_bars <= 5):
            raise ValueError("Invalid target or holding horizon")


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791330879"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(200, params.autocorr_window + 2, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["prior_return"] = out["return"].shift(1)
        out["direction"] = np.sign(out["return"])
        out["lag_direction"] = out["direction"].shift(1)
        up = out["return"].gt(0)
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["sign_autocorr"] = out["direction"].rolling(
            params.autocorr_window, min_periods=params.autocorr_window
        ).corr(out["lag_direction"]).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["entry"] = (
            out["up_streak"].eq(params.streak_length)
            & out["return"].lt(out["prior_return"])
            & out["sign_autocorr"].ge(params.min_autocorr)
            & data["close"].gt(out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this open.
                if held == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                held += 1
                # Close-confirmed targets exit at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.max_hold_bars:
                    active = False
                    continue
                signals[i] = 1
            elif entries[i]:
                active = True
                held = 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], ["high", "close"]] = 102.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
