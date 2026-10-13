from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    budget_bars: int = 2
    percentile_bars: int = 63
    breakout_quantile: float = 0.55


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791199312"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.budget_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.budget_bars < 1 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be positive; percentile_bars must be at least two")
        if not 0.5 < params.breakout_quantile < 1.0:
            raise ValueError("breakout_quantile must be between 0.5 and 1")

        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["prior_budget"] = out["return"].abs().shift(1).rolling(
            params.budget_bars, min_periods=params.budget_bars
        ).sum()
        out["budget_breakout"] = (
            out["return"] / out["prior_budget"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        history = out["budget_breakout"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        )
        out["upper_threshold"] = history.quantile(params.breakout_quantile)
        out["lower_threshold"] = history.quantile(1.0 - params.breakout_quantile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        score = indicators["budget_breakout"]
        upper = indicators["upper_threshold"]
        lower = indicators["lower_threshold"]
        valid = np.isfinite(score) & np.isfinite(upper) & np.isfinite(lower)
        long_entry = valid & (score > upper) & (score > 0.0)
        short_entry = valid & (score < lower) & (score < 0.0)
        events = np.select([long_entry, short_entry], [1, -1], default=0)

        raw = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                # Opposing entry eligibility exits to flat for this decision bar.
                position = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {
            "budget_breakout": [np.nan, 0.8, 0.1, -0.8, -0.9, -0.1, 0.8, 0.9],
            "upper_threshold": 0.2,
            "lower_threshold": -0.2,
        },
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result.index.equals(data.index)
    assert (result["size"] > 0.0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
