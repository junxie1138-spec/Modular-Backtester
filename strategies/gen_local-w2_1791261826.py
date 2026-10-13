from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapVarianceParams:
    lookback: int = 8
    variance_threshold: float = 0.55
    cancellation_threshold: float = 0.25
    min_mean_gap: float = 0.001
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapVarianceParams]):
    strategy_id = "gen_local-w2_1791261826"

    @classmethod
    def params_type(cls) -> type[GapVarianceParams]:
        return GapVarianceParams

    def warmup_bars(self, params: GapVarianceParams) -> int:
        if not 4 <= params.lookback <= 9:
            raise ValueError("lookback must be between 4 and 9")
        if not 15 <= params.holding_bars <= 20:
            raise ValueError("holding_bars must be between 15 and 20")
        if not 0 < params.variance_threshold <= 1:
            raise ValueError("variance_threshold must be in (0, 1]")
        if not 0 <= params.cancellation_threshold < 1:
            raise ValueError("cancellation_threshold must be in [0, 1)")
        if not np.isfinite(params.min_mean_gap) or params.min_mean_gap <= 0:
            raise ValueError("min_mean_gap must be finite and positive")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: GapVarianceParams) -> pd.DataFrame:
        self.warmup_bars(params)
        n = params.lookback
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        mean_abs = gap.abs().rolling(n).mean()
        mean_gap = gap.rolling(n).mean()
        up_energy = gap.clip(lower=0).pow(2).rolling(n).sum()
        down_energy = gap.clip(upper=0).pow(2).rolling(n).sum()
        total_energy = (up_energy + down_energy).replace(0, np.nan)
        result = pd.DataFrame({
            "gap": gap,
            "mean_gap": mean_gap,
            "mean_abs_gap": mean_abs,
            "up_energy": up_energy,
            "down_energy": down_energy,
            "total_energy": total_energy,
            "variance_skew": (up_energy - down_energy) / total_energy,
            "cancellation": mean_gap.abs() / mean_abs.replace(0, np.nan),
            "positive_count": gap.gt(0).rolling(n).sum(),
            "negative_count": gap.lt(0).rolling(n).sum(),
            "ready": gap.rolling(n).count().eq(n),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapVarianceParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            indicators["ready"].astype(bool)
            & indicators["mean_abs_gap"].ge(params.min_mean_gap)
            & indicators["cancellation"].le(params.cancellation_threshold)
            & indicators["variance_skew"].abs().ge(params.variance_threshold)
            & indicators["positive_count"].ge(2)
            & indicators["negative_count"].ge(2)
        )
        candidates = (-np.sign(indicators["variance_skew"])).where(eligible, 0)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(candidates.to_numpy()):
            if position:
                if i - entry_bar == params.holding_bars:
                    position = 0
                    continue
            elif candidate:
                position = int(candidate)
                entry_bar = i
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapVarianceParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=45))
    for side in (-1, 1):
        indicators = pd.DataFrame({
            "ready": True, "mean_abs_gap": 0.002,
            "cancellation": 0.1, "variance_skew": -side * 0.8,
            "positive_count": 3.0, "negative_count": 5.0,
        }, index=data.index)
        signal = strategy.generate_signals(data, indicators, None, params).data["signal"]
        assert signal.iloc[0] == 0
        assert signal.iloc[1:params.holding_bars + 1].eq(side).all()
        assert signal.iloc[params.holding_bars + 1] == 0
        assert signal.iloc[params.holding_bars + 2] == side
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=data.index,
    )
    indicators = strategy.indicators(flat, params)
    assert np.isfinite(indicators.to_numpy(dtype=float)).all()
    assert strategy.generate_signals(flat, indicators, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
