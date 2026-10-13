from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelayParams:
    slope_window: int = 4
    regime_window: int = 20
    percentile_window: int = 63
    entry_quantile: float = 0.7


class GeneratedStrategy(BaseStrategy[RelayParams]):
    strategy_id = "gen_local_1791313446"

    @classmethod
    def params_type(cls):
        return RelayParams

    @staticmethod
    def warmup_bars(params: RelayParams) -> int:
        return int(max(params.slope_window + 1, params.regime_window) + 2 * params.percentile_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RelayParams) -> pd.DataFrame:
        if min(params.slope_window, params.regime_window, params.percentile_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        gap_ratio = data["open"] / data["close"].shift(1)
        day_ratio = data["close"] / data["open"]
        out["gap_return"] = np.log(gap_ratio.where(gap_ratio > 0)).replace([np.inf, -np.inf], np.nan)
        out["day_return"] = np.log(day_ratio.where(day_ratio > 0)).replace([np.inf, -np.inf], np.nan)
        # Mean component returns estimate slopes of latent cumulative session prices.
        out["gap_slope"] = out["gap_return"].rolling(params.slope_window).mean()
        out["day_slope"] = out["day_return"].rolling(params.slope_window).mean()
        out["gap_innovation"] = out["gap_slope"].diff()
        out["day_innovation"] = out["day_slope"].diff()
        out["gap_energy"] = out["gap_return"].pow(2).rolling(params.regime_window).mean()
        out["day_energy"] = out["day_return"].pow(2).rolling(params.regime_window).mean()
        out["total_energy"] = out["gap_energy"] + out["day_energy"]
        out["noise"] = np.sqrt(out["total_energy"].where(out["total_energy"] > 0))
        out["gap_share"] = out["gap_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["regime_threshold"] = out["gap_share"].shift(1).rolling(params.percentile_window).quantile(0.5)
        out["gap_dominant"] = out["gap_share"].gt(out["regime_threshold"])
        ready = out["gap_share"].notna() & out["regime_threshold"].notna()
        out["leader"] = out["day_innovation"].where(out["gap_dominant"], out["gap_innovation"]).where(ready)
        out["follower"] = out["gap_innovation"].where(out["gap_dominant"], out["day_innovation"]).where(ready)
        out["score"] = (out["leader"] - out["follower"]) / out["noise"]
        out["absolute_score"] = out["score"].abs()
        out["entry_threshold"] = out["absolute_score"].shift(1).rolling(params.percentile_window).quantile(params.entry_quantile)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RelayParams) -> SignalFrame:
        entry = indicators["leader"].gt(0) & indicators["follower"].lt(0) & indicators["score"].gt(indicators["entry_threshold"])
        df = pd.DataFrame(index=data.index)
        # Exposure persists exactly while entry remains true; its reversal exits.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RelayParams()
    rng = np.random.default_rng(1791313446)
    index = pd.date_range("2020-01-01", periods=512, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    previous = np.r_[100.0, close[:-1]]
    opening = previous * np.exp(rng.normal(0.0, 0.006, len(index)))
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.002, "low": np.minimum(opening, close) * 0.998, "close": close, "volume": 1000000.0}, index=index)
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    entry = indicators["leader"].gt(0) & indicators["follower"].lt(0) & indicators["score"].gt(indicators["entry_threshold"])
    pd.testing.assert_series_equal(result["signal"], entry.astype(int).shift(1).fillna(0).astype(int), check_names=False)
    pd.testing.assert_frame_equal(indicators.iloc[:300], GeneratedStrategy.indicators(data.iloc[:300], params))
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    assert result["signal"].sum() > 0
    assert result["signal"].iloc[:GeneratedStrategy.warmup_bars(params)].eq(0).all()


if __name__ == "__main__":
    _self_check()
