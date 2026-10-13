from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 4
    area_threshold: float = 0.08
    correlation_ceiling: float = -0.05


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790903539"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(2, min(8, int(params.window))) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        window = max(2, min(8, int(params.window)))
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        x = out["lag_return"]
        y = out["return"]
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, window):
            wedge = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * wedge.rolling(window - lag, min_periods=window - lag).sum()
        out["signature_area"] = area
        out["area_scale"] = x.abs().rolling(window).sum() * y.abs().rolling(window).sum()
        out["normalized_area"] = (area / out["area_scale"].where(out["area_scale"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["autocorrelation"] = x.rolling(window).corr(y).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        # The identical condition controls entry and reversal exit.
        condition = (
            (indicators["return"] < 0)
            & (indicators["normalized_area"] < -abs(params.area_threshold))
            & (indicators["autocorrelation"] < params.correlation_ceiling)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    flat = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert flat["signal"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
    probe = pd.DataFrame({"return": [-0.01, 0.01, -0.01, np.nan], "normalized_area": [-0.2, -0.2, -0.2, np.nan], "autocorrelation": [-0.5, -0.5, -0.5, np.nan]}, index=index[:4])
    result = GeneratedStrategy.generate_signals(data.iloc[:4], probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
