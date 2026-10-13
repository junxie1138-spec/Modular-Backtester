from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SessionContrastParams:
    depth_window: int = 5
    pressure_window: int = 3
    min_depth: float = 0.012
    min_contrast: float = 0.25


class GeneratedStrategy(BaseStrategy[SessionContrastParams]):
    strategy_id = "gen_local_1791235526"

    @classmethod
    def params_type(cls) -> type[SessionContrastParams]:
        return SessionContrastParams

    def warmup_bars(self, params: SessionContrastParams) -> int:
        return int(max(params.depth_window, params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: SessionContrastParams) -> pd.DataFrame:
        if params.depth_window < 2 or params.pressure_window < 1:
            raise ValueError("Lookback windows must be positive; depth_window must be at least two.")
        if not 0.0 < params.min_depth < 1.0 or not 0.0 < params.min_contrast < 1.0:
            raise ValueError("Thresholds must lie strictly between zero and one.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        opening = data["open"].astype(float).replace([np.inf, -np.inf], np.nan)
        opening = opening.where(opening > 0.0)
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0.0)

        out["peak"] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out["trough"] = close.rolling(params.depth_window, min_periods=params.depth_window).min()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["runup"] = close / out["trough"] - 1.0
        out["overnight_return"] = opening / close.shift(1) - 1.0
        out["intraday_return"] = close / opening - 1.0
        out["weighted_contrast"] = volume * (out["intraday_return"] - out["overnight_return"])
        out["weighted_activity"] = volume * (out["intraday_return"].abs() + out["overnight_return"].abs())
        out["contrast_sum"] = out["weighted_contrast"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["activity_sum"] = out["weighted_activity"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["session_contrast"] = out["contrast_sum"] / out["activity_sum"].where(out["activity_sum"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SessionContrastParams,
    ) -> SignalFrame:
        depth = indicators["drawdown"]
        rebound = indicators["runup"]
        contrast = indicators["session_contrast"]
        valid = np.isfinite(depth) & np.isfinite(rebound) & np.isfinite(contrast)
        long_entry = valid & (depth >= params.min_depth) & (depth > rebound) & (contrast >= params.min_contrast)
        short_entry = valid & (rebound >= params.min_depth) & (rebound > depth) & (contrast <= -params.min_contrast)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)

        # Only an opposite qualifying entry exits an existing position.
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SessionContrastParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "drawdown": [0.02, np.nan, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0],
        "runup": [0.0, np.nan, 0.0, 0.02, 0.02, 0.0, 0.0, 0.0],
        "session_contrast": [0.5, np.nan, 0.0, -0.5, -0.5, 0.0, 0.5, 0.0],
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert result.index.equals(index)
    assert (result["size"] > 0.0).all()
    actual = strategy.indicators(data, params)
    assert actual["session_contrast"].isna().all()
    assert strategy.warmup_bars(params) >= max(params.depth_window, params.pressure_window + 1)


if __name__ == "__main__":
    _self_check()
