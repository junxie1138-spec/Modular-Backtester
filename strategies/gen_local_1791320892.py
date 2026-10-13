from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 4
    block_window: int = 3
    shift_threshold: float = 0.9
    occupancy_level: float = 0.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791320892"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.ma_window < 2 or params.block_window < 1:
            raise ValueError("Invalid rolling windows")
        if not np.isfinite(params.shift_threshold) or params.shift_threshold <= 0:
            raise ValueError("shift_threshold must be finite and positive")
        if not np.isfinite(params.occupancy_level) or params.occupancy_level < 0:
            raise ValueError("occupancy_level must be finite and nonnegative")
        bars = params.ma_window + 2 * params.block_window - 1
        if bars > 10:
            raise ValueError("Combined warmup must not exceed ten bars")
        return int(bars)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["recent"] = out["z"].rolling(params.block_window, min_periods=params.block_window).mean()
        out["previous"] = out["recent"].shift(params.block_window)
        out["displacement"] = out["recent"] - out["previous"]
        out["enter"] = (
            (out["previous"] < -params.occupancy_level)
            & (out["recent"] > params.occupancy_level)
            & (out["displacement"] >= params.shift_threshold)
        ).fillna(False)
        out["exit"] = (
            (out["previous"] > params.occupancy_level)
            & (out["recent"] < -params.occupancy_level)
            & (out["displacement"] <= -params.shift_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        events = pd.Series(
            np.where(indicators["enter"].fillna(False), 1.0,
                     np.where(indicators["exit"].fillna(False), 0.0, np.nan)),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    assert strategy.warmup_bars(params) == 9
    assert strategy.warmup_bars(Params(ma_window=5)) == 10
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({column: np.ones(6) for column in ("open", "high", "low", "close", "volume")}, index=index)
    events = pd.DataFrame({"enter": [False, True, False, False, False, False], "exit": [False, False, False, True, False, False]}, index=index)
    result = strategy.generate_signals(data, events, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
