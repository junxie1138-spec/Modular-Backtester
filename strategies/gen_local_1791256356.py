from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankMemoryParams:
    rank_window: int = 63
    memory_window: int = 32
    pressure_window: int = 8
    entry_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[RankMemoryParams]):
    strategy_id = "gen_local_1791256356"

    @classmethod
    def params_type(cls):
        return RankMemoryParams

    def warmup_bars(self, params: RankMemoryParams) -> int:
        if min(params.rank_window, params.memory_window, params.pressure_window) < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.0 < params.entry_percentile <= 1.0:
            raise ValueError("entry_percentile must be in (0, 1].")
        return int(max(200, 2 * params.rank_window + params.memory_window + params.pressure_window + 2))

    def indicators(self, data: pd.DataFrame, params: RankMemoryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["return"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["lagged_rank"] = out["return_rank"].shift(1)
        out["rank_memory"] = out["return_rank"].rolling(params.memory_window, min_periods=params.memory_window).corr(out["lagged_rank"]).clip(-1.0, 1.0).replace([np.inf, -np.inf], np.nan)
        out["rank_pressure"] = out["return_rank"].rolling(params.pressure_window, min_periods=params.pressure_window).mean() - 0.5
        out["forecast"] = out["rank_memory"] * out["rank_pressure"]
        out["forecast_rank"] = out["forecast"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RankMemoryParams,
    ) -> SignalFrame:
        bullish = (
            (data["close"] > indicators["ma200"])
            & (indicators["forecast"] > 0.0)
            & (indicators["forecast_rank"] >= params.entry_percentile)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The composite entry condition reversing is the sole exit.
        df["signal"] = bullish.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RankMemoryParams()
    strategy = GeneratedStrategy()
    t = np.arange(600, dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.0006 + 0.003 * np.sin(t / 5.0) + 0.002 * np.sin(t / 17.0)))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(t), freq="B"),
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, ctx, params).data
    expected = ((data["close"] > ind["ma200"]) & (ind["forecast"] > 0.0) & (ind["forecast_rank"] >= params.entry_percentile)).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(result["signal"], expected, check_names=False)
    assert set(result["signal"].unique()) == {0, 1}
    assert result.index.equals(data.index) and (result["size"] > 0.0).all()
    prefix = data.iloc[:400]
    prefix_ind = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(prefix_ind, ind.iloc[:400])
    pd.testing.assert_frame_equal(strategy.generate_signals(prefix, prefix_ind, ctx, params).data, result.iloc[:400])
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    assert not strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
