from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalCollisionParams:
    pattern_window: int = 4
    collision_jump: float = 0.2
    dominance: float = 0.5


class GeneratedStrategy(BaseStrategy[OrdinalCollisionParams]):
    strategy_id = "gen_local_1791318336"

    @classmethod
    def params_type(cls) -> type[OrdinalCollisionParams]:
        return OrdinalCollisionParams

    @staticmethod
    def warmup_bars(params: OrdinalCollisionParams) -> int:
        if not 2 <= params.pattern_window <= 4:
            raise ValueError("pattern_window must be between 2 and 4")
        if not 0.0 < params.collision_jump <= 1.0:
            raise ValueError("collision_jump must be in (0, 1]")
        if not 0.0 < params.dominance <= 1.0:
            raise ValueError("dominance must be in (0, 1]")
        return int(2 * params.pattern_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalCollisionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        w = params.pattern_window
        out = pd.DataFrame(index=data.index)
        out["old_close"] = data["close"].shift(2)
        out["middle_close"] = data["close"].shift(1)
        a, b, c = out["old_close"], out["middle_close"], data["close"]
        out["valid_pattern"] = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 0) & (b > 0) & (c > 0)
        # Equal closes retain chronological order when assigning ordinal codes.
        out["pattern"] = ((a > b).astype(int) * 4 + (a > c).astype(int) * 2 + (b > c).astype(int)).where(out["valid_pattern"])
        out["ordinal_vote"] = np.sign(c - a).where(out["valid_pattern"])
        out["valid_count"] = out["valid_pattern"].astype(int).rolling(w, min_periods=w).sum()
        collision_pairs = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            column = "count_" + str(code)
            out[column] = out["pattern"].eq(code).astype(float).rolling(w, min_periods=w).sum()
            collision_pairs = collision_pairs + out[column] * (out[column] - 1.0)
        out["collision"] = (collision_pairs / (w * (w - 1))).where(out["valid_count"].eq(w))
        out["previous_collision"] = out["collision"].shift(w)
        out["collision_gain"] = out["collision"] - out["previous_collision"]
        out["direction"] = out["ordinal_vote"].rolling(w, min_periods=w).mean()
        out["changepoint"] = out["collision_gain"].ge(params.collision_jump).fillna(False)
        bullish = out["changepoint"] & out["direction"].ge(params.dominance)
        bearish = out["changepoint"] & out["direction"].le(-params.dominance)
        out["regime_event"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalCollisionParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        # A bullish changepoint activates the entry condition; a bearish one flips it off.
        regime = indicators["regime_event"].replace(0, np.nan).ffill().fillna(0)
        df["signal"] = regime.gt(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = OrdinalCollisionParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [1.0] * 6}, index=index)
    events = pd.DataFrame({"regime_event": [0, 1, 0, 0, -1, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, events, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 10
    flat = GeneratedStrategy.indicators(data, params)
    assert flat["regime_event"].eq(0).all()


if __name__ == "__main__":
    _self_check()
