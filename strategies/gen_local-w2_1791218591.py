from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankShiftParams:
    window: int = 4
    threshold: float = 0.375


class GeneratedStrategy(BaseStrategy[RankShiftParams]):
    strategy_id = "gen_local-w2_1791218591"

    @classmethod
    def params_type(cls) -> type[RankShiftParams]:
        return RankShiftParams

    def warmup_bars(self, params: RankShiftParams) -> int:
        return 2 * params.window + 1

    def indicators(self, data: pd.DataFrame, params: RankShiftParams) -> pd.DataFrame:
        if params.window < 2 or not 0.0 < params.threshold < 1.0:
            raise ValueError("window must be >= 2 and threshold must be between 0 and 1")
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        dominance = pd.Series(0.0, index=data.index)
        for recent in range(params.window):
            for previous in range(params.window):
                dominance += np.sign(returns.shift(recent) - returns.shift(params.window + previous))
        dominance /= float(params.window ** 2)
        return pd.DataFrame({"returns": returns, "dominance": dominance}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RankShiftParams,
    ) -> SignalFrame:
        score = indicators["dominance"]
        # Mirrored thresholds change position; neutral or missing evidence preserves it.
        decisions = np.select(
            [score.gt(params.threshold), score.lt(-params.threshold)],
            [1.0, 0.0],
            default=np.nan,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(decisions, index=data.index).ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RankShiftParams()
    index = pd.date_range("2020-01-01", periods=13)
    returns = pd.Series([0.0] + [-0.004] * 4 + [0.004] * 4 + [-0.004] * 4, index=index)
    close = 100.0 * (1.0 + returns).cumprod()
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    assert strategy.warmup_bars(params) == 9
    assert np.isclose(derived["dominance"].iloc[8], 1.0)
    assert np.isclose(derived["dominance"].iloc[12], -1.0)
    assert signals["signal"].iloc[8:13].tolist() == [0, 1, 1, 1, 0]
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
