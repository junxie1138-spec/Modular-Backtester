from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakResonanceParams:
    streak_length: int = 3
    correlation_window: int = 60
    correlation_floor: float = 0.20
    hold_bars: int = 7

    def __post_init__(self):
        if not 1 <= self.streak_length <= 8:
            raise ValueError("streak_length must be between 1 and 8")
        if self.correlation_window < 10 or self.hold_bars < 1:
            raise ValueError("correlation_window must be >= 10 and hold_bars >= 1")
        if not 0.0 < self.correlation_floor < 1.0:
            raise ValueError("correlation_floor must be between 0 and 1")


class GeneratedStrategy(BaseStrategy[StreakResonanceParams]):
    strategy_id = "gen_local_1790900312"

    @classmethod
    def params_type(cls) -> type[StreakResonanceParams]:
        return StreakResonanceParams

    @staticmethod
    def warmup_bars(params: StreakResonanceParams) -> int:
        # Eight-bar streak cap, five-bar lag, and one bar for the close difference.
        return int(params.correlation_window + 8 + 5 + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakResonanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["change"] = close.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["direction"] = np.sign(out["change"]).astype(int)
        groups = out["direction"].ne(out["direction"].shift(1)).cumsum()
        out["streak"] = (out.groupby(groups).cumcount() + 1).where(
            out["direction"].ne(0), 0
        ).clip(upper=8)
        out["signed_streak"] = out["streak"] * out["direction"]
        out["lagged_streak"] = out["signed_streak"].shift(5)
        out["correlation"] = out["signed_streak"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(out["lagged_streak"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        ready = np.arange(len(data)) >= GeneratedStrategy.warmup_bars(params) - 1
        eligible = ready & out["streak"].eq(params.streak_length) & out["correlation"].ge(
            params.correlation_floor
        )
        out["entry"] = out["direction"].where(eligible, 0).astype(int)
        out["strength_size"] = 0.20 + 0.75 * out["correlation"].clip(0.0, 1.0)
        return out.fillna(0.0)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakResonanceParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength_size"].fillna(0.20).to_numpy(dtype=float)
        position, exit_at, entry_size = 0, -1, 0.20
        for i in range(len(data)):
            if position != 0 and i == exit_at:
                position, entry_size = 0, 0.20
            elif position == 0 and entries[i] != 0:
                position = int(entries[i])
                entry_size = float(np.clip(strengths[i], 0.20, 0.95))
                exit_at = i + params.hold_bars
            signals[i], sizes[i] = position, entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StreakResonanceParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(index) and np.isfinite(derived.to_numpy()).all()
    assert not derived["entry"].any()
    supplied = pd.DataFrame(
        {"entry": [1] + [-1] * 23, "strength_size": np.linspace(0.30, 0.90, 24)},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, supplied, None, params).data
    assert result["signal"].tolist()[:10] == [0] + [1] * 7 + [0, -1]
    assert result["signal"].tolist()[9:17] == [-1] * 7 + [0]
    assert np.allclose(result["size"].iloc[1:8], 0.30)
    assert result["size"].gt(0).all()
    close = pd.Series(100.0 + np.arange(160) + 0.1 * np.sin(np.arange(160)), index=pd.date_range("2021-01-01", periods=160))
    rising = pd.DataFrame({"close": close})
    assert GeneratedStrategy.indicators(rising, params)["streak"].iloc[-1] == 8
    prefix = GeneratedStrategy.indicators(rising.iloc[:100], params)
    full = GeneratedStrategy.indicators(rising, params)
    pd.testing.assert_frame_equal(prefix, full.iloc[:100])


if __name__ == "__main__":
    _self_check()
