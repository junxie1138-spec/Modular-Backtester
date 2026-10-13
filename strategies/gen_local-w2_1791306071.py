from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    ma_window: int = 42
    smoothing_window: int = 10
    regime_window: int = 63
    z_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = "gen_local-w2_1791306071"

    @classmethod
    def params_type(cls) -> type[CycleParams]:
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(params.ma_window + params.smoothing_window - 1, params.regime_window + 2) + 1)

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        if min(params.ma_window, params.regime_window) < 2 or params.smoothing_window < 1:
            raise ValueError("MA and regime windows must be at least two; smoothing must be positive")
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError("z_threshold must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["smoothed_z"] = out["z"].rolling(params.smoothing_window, min_periods=params.smoothing_window).mean()
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["autocorrelation"] = out["return"].rolling(params.regime_window, min_periods=params.regime_window).corr(out["lagged_return"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CycleParams) -> SignalFrame:
        z = indicators["smoothed_z"]
        correlation = indicators["autocorrelation"]
        valid = np.isfinite(z) & np.isfinite(correlation)
        continuation = correlation > 0.0
        bullish = valid & ((continuation & (z > params.z_threshold)) | (~continuation & (z < -params.z_threshold)))
        confirmed = bullish & bullish.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        # Once entered, confirmation stays true until the bullish entry predicate flips.
        df["signal"] = confirmed.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CycleParams()
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"smoothed_z": [np.nan, 0.8, 0.8, 0.8, 0.0, -0.8, -0.8, -0.8, 0.8], "autocorrelation": [np.nan, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 65
    truncated = strategy.generate_signals(data.iloc[:7], indicators.iloc[:7], None, params).data
    pd.testing.assert_frame_equal(result.iloc[:7], truncated)


if __name__ == "__main__":
    _self_check()
