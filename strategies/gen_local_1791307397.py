from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 3
    reference_bars: int = 252
    percentile: float = 0.8


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local_1791307397"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    @staticmethod
    def warmup_bars(params: SignatureSeasonParams) -> int:
        return int(params.path_bars + params.reference_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        if params.path_bars < 2 or params.reference_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.percentile < 1.0:
            raise ValueError("percentile must be between zero and one")
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        gap = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        intraday = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature; older increments precede newer ones.
        for older in range(1, params.path_bars):
            for newer in range(older):
                area += 0.5 * (
                    gap.shift(older) * intraday.shift(newer)
                    - intraday.shift(older) * gap.shift(newer)
                )
        gap_mass = gap.abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        intraday_mass = intraday.abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        scale = gap_mass * intraday_mass
        area = (area / scale.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        threshold = area.shift(1).rolling(
            params.reference_bars, min_periods=params.reference_bars
        ).quantile(params.percentile)
        seasonal = pd.Series(data.index.dayofweek.isin([1, 2]), index=data.index)
        return pd.DataFrame({
            "gap": gap,
            "intraday": intraday,
            "gap_mass": gap_mass,
            "intraday_mass": intraday_mass,
            "scale": scale,
            "area": area,
            "threshold": threshold,
            "seasonal": seasonal,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureSeasonParams,
    ) -> SignalFrame:
        condition = (
            indicators["seasonal"].fillna(False)
            & indicators["area"].notna()
            & indicators["threshold"].notna()
            & indicators["area"].gt(indicators["threshold"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Target exposure becomes zero exactly when the joint entry condition flips off.
        df["signal"] = condition.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureSeasonParams()
    index = pd.bdate_range("2020-01-01", periods=600)
    t = np.arange(len(index), dtype=float)
    close = pd.Series(100.0 * np.exp(0.0002 * t + 0.01 * np.sin(t)), index=index)
    opening = close.shift(1).fillna(close.iloc[0]) * np.exp(0.005 * np.cos(t))
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.001,
                         "low": np.minimum(opening, close) * 0.999,
                         "close": close, "volume": 1000000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    pd.testing.assert_frame_equal(derived.iloc[:450], GeneratedStrategy.indicators(data.iloc[:450], params))
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected = (derived["seasonal"] & derived["area"].gt(derived["threshold"])).astype(int)
    assert result["signal"].equals(expected.shift(1).fillna(0).astype(int))
    assert result["signal"].isin([0, 1]).all()
    assert not result["signal"].rolling(3).sum().eq(3).any()
    assert result["signal"].iloc[:GeneratedStrategy.warmup_bars(params)].eq(0).all()
    assert result["size"].gt(0).all()
    forced = derived.copy()
    forced["area"] = 1.0
    forced["threshold"] = 0.5
    forced_result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    assert forced_result["signal"].equals(forced["seasonal"].astype(int).shift(1).fillna(0).astype(int))


if __name__ == "__main__":
    _self_check()
