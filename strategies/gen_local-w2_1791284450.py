from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReleaseParams:
    lookback: int = 20
    gap_z: float = 1.5
    muted_ratio: float = 0.35
    acceleration: float = 1.1
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapReleaseParams]):
    strategy_id = "gen_local-w2_1791284450"

    @classmethod
    def params_type(cls) -> type[GapReleaseParams]:
        return GapReleaseParams

    @staticmethod
    def warmup_bars(params: GapReleaseParams) -> int:
        return max(params.lookback, 3) + 4

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapReleaseParams) -> pd.DataFrame:
        if params.lookback < 2 or params.gap_z <= 0:
            raise ValueError("lookback must be >= 2 and gap_z positive")
        if not 0 < params.muted_ratio < 1 or params.acceleration < 1:
            raise ValueError("muted_ratio must be in (0, 1) and acceleration >= 1")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["gap"] = opening.div(close.shift(1)).sub(1)
        out["prior_vol"] = out["return"].rolling(
            params.lookback, min_periods=params.lookback
        ).std(ddof=0).shift(1)
        out["prior_drift"] = out["return"].rolling(3, min_periods=3).sum().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)

        shock = (
            out["prior_vol"].gt(0)
            & out["gap"].abs().ge(params.gap_z * out["prior_vol"])
            & out["gap"].mul(out["prior_drift"]).lt(0)
            & out["return"].abs().le(params.muted_ratio * out["gap"].abs())
        )
        out["shock_direction"] = np.sign(out["gap"]).where(shock, 0).fillna(0)
        out["pending_direction"] = out["shock_direction"].shift(2).fillna(0)
        out["first_confirmation_return"] = out["return"].shift(1)
        confirmed = (
            out["pending_direction"].ne(0)
            & out["first_confirmation_return"].mul(out["pending_direction"]).gt(0)
            & out["return"].mul(out["pending_direction"]).gt(0)
            & out["return"].abs().ge(
                params.acceleration * out["first_confirmation_return"].abs()
            )
        )
        out["entry_direction"] = out["pending_direction"].where(confirmed, 0).fillna(0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapReleaseParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        candidates = indicators["entry_direction"].reindex(data.index).fillna(0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered = -1
        for i, candidate in enumerate(candidates):
            if position:
                if i - entered >= params.hold_bars:
                    position = 0
            elif candidate:
                position = int(candidate)
                entered = i
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    for hold in (3, 4, 5):
        candidates = [1] + [-1] * hold + [-1] + [1] * (hold + 2)
        index = pd.date_range("2020-01-01", periods=len(candidates), freq="B")
        data = pd.DataFrame(index=index)
        indicators = pd.DataFrame({"entry_direction": candidates}, index=index)
        result = GeneratedStrategy.generate_signals(
            data, indicators, None, GapReleaseParams(hold_bars=hold)
        ).data
        values = result["signal"].tolist()
        assert values[0] == 0
        assert values[1:hold + 1] == [1] * hold
        assert values[hold + 1] == 0
        assert values[hold + 2:2 * hold + 2] == [-1] * hold
        assert values[2 * hold + 2] == 0
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
