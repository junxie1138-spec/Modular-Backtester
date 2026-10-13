from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapOutbreakParams:
    gap_window: int = 12
    noise_window: int = 20
    gap_floor: float = 0.35
    max_incidence: float = 0.25
    breakout_multiplier: float = 1.05


class GeneratedStrategy(BaseStrategy[GapOutbreakParams]):
    strategy_id = "gen_local_1790925112"

    @classmethod
    def params_type(cls):
        return GapOutbreakParams

    def warmup_bars(self, params: GapOutbreakParams) -> int:
        return int(max(200, params.noise_window + params.gap_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapOutbreakParams) -> pd.DataFrame:
        if params.gap_window < 1 or params.noise_window < 1:
            raise ValueError("Lookbacks must be positive.")
        if not np.isfinite([params.gap_floor, params.max_incidence, params.breakout_multiplier]).all():
            raise ValueError("Thresholds must be finite.")
        if params.gap_floor <= 0 or not 0 <= params.max_incidence <= 1 or params.breakout_multiplier < 1:
            raise ValueError("Invalid gap thresholds.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = np.maximum(
            data["high"] - data["low"],
            np.maximum(
                (data["high"] - out["prior_close"]).abs(),
                (data["low"] - out["prior_close"]).abs(),
            ),
        )
        out["prior_atr"] = out["true_range"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).mean().shift(1).replace(0.0, np.nan)
        out["gap"] = (data["open"] - out["prior_close"]) / out["prior_atr"]
        out["prior_gap_peak"] = out["gap"].clip(lower=0.0).rolling(
            params.gap_window, min_periods=params.gap_window
        ).max().shift(1)
        out["gap_seed"] = out["gap"].gt(params.gap_floor).astype(float).where(out["gap"].notna())
        out["prior_incidence"] = out["gap_seed"].rolling(
            params.gap_window, min_periods=params.gap_window
        ).mean().shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["gap_threshold"] = (out["prior_gap_peak"] * params.breakout_multiplier).clip(
            lower=params.gap_floor
        )
        out["ready"] = (
            np.isfinite(out[["gap", "gap_threshold", "prior_incidence", "ma200"]]).all(axis=1)
            & np.isfinite(data[["open", "close"]]).all(axis=1)
        )
        out["eligible"] = (
            out["ready"]
            & out["gap"].gt(out["gap_threshold"])
            & out["prior_incidence"].le(params.max_incidence)
            & data["close"].ge(data["open"])
            & data["close"].gt(out["ma200"])
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapOutbreakParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # A false entry condition reverses the position to flat; no timed exit.
        df["signal"] = indicators["eligible"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo():
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=204),
    )
    data.iloc[200:204, :4] = [
        [102, 104, 101, 103],
        [106, 108, 105, 107],
        [107, 108, 106, 107],
        [107, 108, 106, 107],
    ]
    strategy = GeneratedStrategy()
    params = GapOutbreakParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert signals.index.equals(data.index)
    assert signals["signal"].iloc[:201].eq(0).all()
    assert signals["signal"].iloc[201:].tolist() == [1, 1, 0]
    assert signals["size"].gt(0).all()
    prefix = data.iloc[:202]
    earlier = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(earlier, signals.iloc[:202])


if __name__ == "__main__":
    demo()
