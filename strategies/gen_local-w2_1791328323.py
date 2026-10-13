from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSeparationParams:
    ma_bars: int = 4
    scale_bars: int = 12
    clearance_z: float = 0.15


class GeneratedStrategy(BaseStrategy[RangeSeparationParams]):
    strategy_id = "gen_local-w2_1791328323"

    @classmethod
    def params_type(cls) -> type[RangeSeparationParams]:
        return RangeSeparationParams

    def warmup_bars(self, params: RangeSeparationParams) -> int:
        return int(max(params.ma_bars, params.scale_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeSeparationParams) -> pd.DataFrame:
        if params.ma_bars < 2 or params.scale_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not np.isfinite(params.clearance_z) or params.clearance_z <= 0:
            raise ValueError("clearance_z must be finite and positive")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean().shift(1)
        out["scale"] = close.rolling(params.scale_bars, min_periods=params.scale_bars).std(ddof=0).shift(1)
        scale = out["scale"].where(out["scale"].gt(0))
        out["z_low"] = (data["low"] - out["ma"]) / scale
        out["z_high"] = (data["high"] - out["ma"]) / scale
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeSeparationParams) -> SignalFrame:
        valid = np.isfinite(indicators["z_low"]) & np.isfinite(indicators["z_high"])
        enter = valid & indicators["z_low"].gt(params.clearance_z)
        exit_long = valid & indicators["z_high"].lt(-params.clearance_z)
        events = pd.Series(np.select([enter, exit_long], [1.0, 0.0], default=np.nan), index=data.index)
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeSeparationParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"z_low": [np.nan, 0.4, 0.0, np.nan, -0.8, 0.0, 0.5, -0.9], "z_high": [np.nan, 0.8, 0.6, np.nan, -0.4, 0.5, 0.9, -0.5]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 13


if __name__ == "__main__":
    _self_check()
