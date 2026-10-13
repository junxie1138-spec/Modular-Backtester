from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRankParams:
    horizon: int = 18
    rank_window: int = 126
    history_window: int = 504
    min_observations: int = 12
    threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[SeasonalRankParams]):
    strategy_id = "gen_local_1791221888"

    @classmethod
    def params_type(cls):
        return SeasonalRankParams

    def warmup_bars(self, params: SeasonalRankParams) -> int:
        return int(params.horizon + params.rank_window + params.history_window + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalRankParams) -> pd.DataFrame:
        if (params.horizon < 1 or params.rank_window < 2
                or params.history_window < params.min_observations
                or params.min_observations < 2
                or not 0.0 < params.threshold < 0.5):
            raise ValueError("Invalid seasonal rank parameters")
        out = pd.DataFrame(index=data.index)
        out["outcome_return"] = data["close"].pct_change(
            periods=params.horizon, fill_method=None
        ).replace([np.inf, -np.inf], np.nan)
        out["outcome_rank"] = out["outcome_return"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["rank_surprise"] = (
            out["outcome_rank"]
            - (params.rank_window + 1.0) / (2.0 * params.rank_window)
        )
        month = pd.Series(data.index.month, index=data.index)
        # A realized outcome belongs to its entry month's calendar bucket.
        out["origin_month"] = month.shift(params.horizon)
        out["seasonal_score"] = np.nan
        for m in range(1, 13):
            eligible = out["origin_month"].eq(m) & out["rank_surprise"].notna()
            count = eligible.astype(float).rolling(
                params.history_window, min_periods=params.history_window
            ).sum()
            total = out["rank_surprise"].where(eligible, 0.0).rolling(
                params.history_window, min_periods=params.history_window
            ).sum()
            out[f"count_{m}"] = count
            out[f"surprise_sum_{m}"] = total
            out[f"premium_{m}"] = total.div(
                count.where(count >= params.min_observations)
            )
            current = month.eq(m)
            out.loc[current, "seasonal_score"] = out.loc[current, f"premium_{m}"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalRankParams,
    ) -> SignalFrame:
        score = indicators["seasonal_score"]
        direction = np.select(
            [score.gt(params.threshold), score.lt(-params.threshold)],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, raw in enumerate(direction):
            if position:
                if raw == -position:
                    position = 0
            elif raw != 0 and raw == previous:
                position = int(raw)
            signals[i] = position
            previous = int(raw)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SeasonalRankParams()
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"seasonal_score": [0.2, 0.2, 0.0, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2]},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
