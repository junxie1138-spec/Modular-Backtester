from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory: int = 18
    streak_cap: int = 4
    reversal_threshold: float = 0.12
    vol_window: int = 20
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791306781"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.memory + params.streak_cap + 1, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (
            params.memory < 1 or params.streak_cap < 1 or params.vol_window < 2
            or not 0.0 < params.reversal_threshold <= 1.0
            or not np.isfinite(params.target_vol) or params.target_vol <= 0.0
        ):
            raise ValueError("Invalid strategy parameters")
        high_change = data["high"].diff()
        low_change = data["low"].diff()
        migration = pd.Series(
            np.select(
                [(high_change > 0) & (low_change > 0),
                 (high_change < 0) & (low_change < 0)],
                [1, -1], default=0
            ),
            index=data.index, dtype=int
        )
        groups = migration.ne(migration.shift()).cumsum()
        streak = (migration.groupby(groups).cumcount() + 1).clip(
            upper=params.streak_cap
        ) * migration
        score = (
            streak.rolling(params.memory, min_periods=params.memory).sum()
            / (params.memory * params.streak_cap)
        ).fillna(0.0)
        close = data["close"].where(
            (data["close"] > 0) & np.isfinite(data["close"])
        )
        returns = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        raw_vol = returns.rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0) * np.sqrt(252.0)
        annual_vol = raw_vol.clip(lower=0.01).ffill().fillna(0.01)
        ready = (
            (np.arange(len(data)) >= self.warmup_bars(params) - 1)
            & raw_vol.notna()
        )
        return pd.DataFrame({
            "migration": migration,
            "signed_streak": streak,
            "score": score,
            "annual_vol": annual_vol,
            "ready": ready,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        score = indicators["score"].fillna(0.0)
        candidates = np.select(
            [score >= params.reversal_threshold,
             score <= -params.reversal_threshold],
            [1, -1], default=0
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if bool(indicators["ready"].iloc[i]):
                if position and candidate == -position:
                    position = 0
                elif position == 0 and candidate:
                    position = int(candidate)
            positions[i] = position
        df = pd.DataFrame({"signal": positions}, index=data.index)
        confidence = (df["signal"] * score).clip(lower=0.0, upper=1.0)
        size = (
            params.target_vol / indicators["annual_vol"].clip(lower=0.01)
            * (0.25 + 0.75 * confidence)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95)
        df["size"] = size.shift(1).fillna(0.05).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    close = 100.0 + np.arange(100, dtype=float)
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000000.0
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert np.isfinite(derived.to_numpy(dtype=float)).all()
    actual = strategy.generate_signals(data, derived, None, params).data
    warmup = strategy.warmup_bars(params)
    assert (actual["signal"].iloc[:warmup] == 0).all()
    assert actual["signal"].iloc[warmup] == 1
    prefix = strategy.indicators(data.iloc[:60], params)
    pd.testing.assert_frame_equal(prefix, derived.iloc[:60])
    manual = pd.DataFrame({
        "score": [0.5, 0.0, -0.5, -0.5, 0.0, 0.5, 0.5],
        "annual_vol": 0.2, "ready": True
    }, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], manual, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()
    assert result["size"].iloc[1] > result["size"].iloc[2]


if __name__ == "__main__":
    _self_check()
