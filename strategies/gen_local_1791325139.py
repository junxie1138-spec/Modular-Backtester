from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 5
    memory_days: int = 4
    percentile_window: int = 126
    entry_percentile: float = 0.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791325139"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 2 * params.trend_window + params.memory_days + params.percentile_window

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.trend_window, params.memory_days, params.percentile_window) < 2:
            raise ValueError("All windows must be at least two bars.")
        if not 0.0 < params.entry_percentile < 0.5:
            raise ValueError("entry_percentile must be between zero and 0.5.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        out["latent_level"] = out["log_close"].rolling(params.trend_window).mean()
        out["latent_slope"] = out["latent_level"].diff()
        out["predicted_slope"] = out["latent_slope"].shift(1).rolling(params.trend_window).mean()
        out["slope_innovation"] = out["latent_slope"] - out["predicted_slope"]
        out["innovation_debt"] = out["slope_innovation"].rolling(params.memory_days).sum()
        out["entry_threshold"] = out["innovation_debt"].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        # The same predicate controls entry and continued ownership.
        # Its reversal is the only exit; memory_days targets a 3-5 session cycle.
        valid = indicators[["innovation_debt", "entry_threshold"]].notna().all(axis=1)
        entry = valid & (indicators["innovation_debt"] < indicators["entry_threshold"])
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    close = 100.0 * np.exp(0.0002 * np.arange(400) + 0.02 * np.sin(np.arange(400) / 4.0))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    assert indicators["entry_threshold"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    assert strategy.indicators(data.iloc[:250], params).equals(indicators.iloc[:250])
    probe = pd.DataFrame({"innovation_debt": [np.nan, -2.0, -3.0, 0.0, -2.0, 0.0, 0.0], "entry_threshold": [-1.0] * 7}, index=index[:7])
    signals = strategy.generate_signals(data.iloc[:7], probe, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0, 1, 0]
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
