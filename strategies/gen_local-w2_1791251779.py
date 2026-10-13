from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDepletionParams:
    wave_window: int = 4
    rank_window: int = 63
    depletion_threshold: float = 0.30
    min_wave_rank: float = 0.60
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapDepletionParams]):
    strategy_id = "gen_local-w2_1791251779"

    @classmethod
    def params_type(cls) -> type[GapDepletionParams]:
        return GapDepletionParams

    @staticmethod
    def warmup_bars(params: GapDepletionParams) -> int:
        return int(params.rank_window + 2 * params.wave_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapDepletionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["down_gap"] = (-out["gap"]).clip(lower=0.0)
        out["recent_pressure"] = out["down_gap"].rolling(params.wave_window, min_periods=params.wave_window).sum()
        out["prior_pressure"] = out["recent_pressure"].shift(params.wave_window)
        out["reproduction"] = out["recent_pressure"] / out["prior_pressure"].where(out["prior_pressure"] > 0.0)
        out["reproduction_rank"] = out["reproduction"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["prior_wave_rank"] = out["prior_pressure"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDepletionParams,
    ) -> SignalFrame:
        rank = indicators["reproduction_rank"]
        wave_rank = indicators["prior_wave_rank"]
        entry_condition = (
            rank.notna()
            & wave_rank.notna()
            & (rank <= params.depletion_threshold)
            & (wave_rank >= params.min_wave_rank)
            & (indicators["reproduction"] < 1.0)
        )
        conviction = ((params.depletion_threshold - rank) / params.depletion_threshold).clip(0.0, 1.0)
        conviction = (conviction * wave_rank).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        # The position remains active exactly while the entry condition holds.
        # Its reversal is the sole exit; there is no timed or price-based exit.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["size"] = params.min_size + (params.max_size - params.min_size) * conviction
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapDepletionParams()
    t = np.arange(320, dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.001 * np.sin(t / 9.0)))
    previous = np.concatenate(([close[0]], close[:-1]))
    opening = previous * (1.0 + 0.008 * np.sin(t / 7.0))
    data = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * 1.01,
        "low": np.minimum(opening, close) * 0.99,
        "close": close,
        "volume": np.full(len(t), 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=len(t), freq="B"))
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    expected = (
        (indicators["reproduction_rank"] <= params.depletion_threshold)
        & (indicators["prior_wave_rank"] >= params.min_wave_rank)
        & (indicators["reproduction"] < 1.0)
    ).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert set(result["signal"].unique()) <= {0, 1}
    assert np.isfinite(result["size"]).all() and (result["size"] > 0.0).all()
    prefix = GeneratedStrategy.indicators(data.iloc[:240], params)
    pd.testing.assert_frame_equal(indicators.iloc[:240], prefix)


if __name__ == "__main__":
    _self_check()
