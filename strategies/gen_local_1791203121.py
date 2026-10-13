from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDisagreementParams:
    gap_window: int = 4
    disagreement: float = 0.2


class GeneratedStrategy(BaseStrategy[GapDisagreementParams]):
    strategy_id = "gen_local_1791203121"

    @classmethod
    def params_type(cls) -> type[GapDisagreementParams]:
        return GapDisagreementParams

    def warmup_bars(self, params: GapDisagreementParams) -> int:
        return max(2, int(params.gap_window)) + 2

    def indicators(self, data: pd.DataFrame, params: GapDisagreementParams) -> pd.DataFrame:
        window = max(2, int(params.gap_window))
        prior_close = data["close"].shift(1)
        gap = data["open"].div(prior_close.where(prior_close > 0)).sub(1)
        gap = gap.replace([np.inf, -np.inf], np.nan)
        frequency = np.sign(gap).rolling(window, min_periods=window).mean()
        net_gap = gap.rolling(window, min_periods=window).sum()
        absolute_gap = gap.abs().rolling(window, min_periods=window).sum()
        magnitude_balance = net_gap.div(absolute_gap.where(absolute_gap > 0))
        magnitude_balance = magnitude_balance.where(absolute_gap > 0, 0.0)
        score = frequency - magnitude_balance
        return pd.DataFrame({
            "gap": gap,
            "gap_frequency": frequency,
            "net_gap": net_gap,
            "absolute_gap": absolute_gap,
            "magnitude_balance": magnitude_balance,
            "disagreement_score": score,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapDisagreementParams) -> SignalFrame:
        threshold = float(params.disagreement)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("disagreement must be finite and positive")
        score = indicators["disagreement_score"]
        candidate = pd.Series(np.select(
            [score.gt(threshold), score.lt(-threshold)],
            [1, -1], default=0,
        ), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (direction, entry) in enumerate(zip(candidate.to_numpy(), confirmed.to_numpy())):
            if position != 0 and direction == -position:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = GapDisagreementParams()
    indicators = pd.DataFrame({"disagreement_score":
        [np.nan, 0.3, 0.3, 0.0, -0.3, -0.3, -0.3, 0.3, 0.3]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert actual["disagreement_score"].iloc[:4].isna().all()
    assert actual["disagreement_score"].iloc[4:].eq(0).all()


if __name__ == "__main__":
    _self_check()
