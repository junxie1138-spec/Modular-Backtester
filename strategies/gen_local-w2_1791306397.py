from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeakestLinkParams:
    floor_window: int = 3
    rank_window: int = 63
    entry_percentile: float = 0.85
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WeakestLinkParams]):
    strategy_id = "gen_local-w2_1791306397"

    @classmethod
    def params_type(cls) -> type[WeakestLinkParams]:
        return WeakestLinkParams

    @staticmethod
    def warmup_bars(params: WeakestLinkParams) -> int:
        if params.floor_window < 1 or params.rank_window < 2:
            raise ValueError("Invalid rolling windows")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between zero and one")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        return int(params.floor_window + params.rank_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: WeakestLinkParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        daily_return = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        return_floor = daily_return.rolling(
            params.floor_window, min_periods=params.floor_window
        ).min()
        floor_percentile = return_floor.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        return pd.DataFrame({
            "daily_return": daily_return,
            "return_floor": return_floor,
            "floor_percentile": floor_percentile,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WeakestLinkParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        eligible = (
            indicators["return_floor"].gt(0.0)
            & indicators["floor_percentile"].ge(params.entry_percentile)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(eligible):
            if i < exit_at:
                signal[i] = 1
            elif i > exit_at and enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
            # At exit_at, remain flat even if another entry qualifies.
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = WeakestLinkParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "return_floor": [0.01] * 7,
        "floor_percentile": [0.95] * 7,
    }, index=index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0]),
        (2, [0, 1, 1, 0, 1, 1, 0]),
    ):
        params.hold_bars = hold
        result = GeneratedStrategy.generate_signals(data, indicators, None, params)
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0.0).all()
    indicators["floor_percentile"] = np.nan
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 66


if __name__ == "__main__":
    _self_check()
