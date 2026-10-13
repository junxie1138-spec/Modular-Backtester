from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    lookback: int = 40
    atr_period: int = 14
    forecast_threshold: float = 0.2
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local-w2_1791260123"

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        if params.lookback < 3 or params.atr_period < 1 or params.max_hold < 1:
            raise ValueError("Lookback must be at least 3; ATR period and holding limit must be positive.")
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError("Forecast threshold must be finite and positive.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("ATR multiplier must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        denominator = out["prior_close"].where(out["prior_close"] > 0)
        out["gap"] = data["open"] / denominator - 1.0
        out["day_return"] = data["close"] / denominator - 1.0
        out["gap_sign"] = np.sign(out["gap"])
        out["retained_gap"] = out["gap_sign"] * np.minimum(
            out["gap"].abs(),
            (out["gap_sign"] * out["day_return"]).clip(lower=0.0),
        )
        out["lagged_retained"] = out["retained_gap"].shift(1)
        length = params.lookback
        gap_window = out["gap"].rolling(length, min_periods=length)
        retained_window = out["lagged_retained"].rolling(length, min_periods=length)
        out["gap_mean"] = gap_window.mean()
        out["retained_mean"] = retained_window.mean()
        out["retained_variance"] = retained_window.var(ddof=0)
        out["gap_covariance"] = gap_window.cov(out["lagged_retained"], ddof=0)
        out["beta"] = out["gap_covariance"] / out["retained_variance"].where(
            out["retained_variance"] > 0.0
        )
        out["gap_std"] = gap_window.std(ddof=0)
        out["forecast"] = out["gap_mean"] + out["beta"] * (
            out["retained_gap"] - out["retained_mean"]
        )
        out["score"] = out["forecast"] / out["gap_std"].where(out["gap_std"] > 0.0)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["prior_close"]).abs(),
                (data["low"] - out["prior_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        persistent = indicators["beta"].gt(0.0)
        long_setup = (
            persistent
            & indicators["score"].ge(params.forecast_threshold)
            & indicators["retained_gap"].gt(0.0)
        ).fillna(False)
        short_setup = (
            persistent
            & indicators["score"].le(-params.forecast_threshold)
            & indicators["retained_gap"].lt(0.0)
        ).fillna(False)
        confirmed = np.where(
            long_setup & long_setup.shift(1, fill_value=False),
            1,
            np.where(short_setup & short_setup.shift(1, fill_value=False), -1, 0),
        )
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = -np.inf
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        for i, (price, volatility) in enumerate(zip(closes, atr)):
            if not np.isfinite(price) or not np.isfinite(volatility) or volatility <= 0.0:
                position = 0
            elif position:
                # Initialize from the first close after the next-open entry.
                high_water = max(high_water, price)
                # Preserve the ratchet even when ATR subsequently expands.
                stop = max(stop, high_water - params.trail_atr * volatility)
                # The specified close-high trail applies to both directions:
                # it protects longs and harvests downward moves for shorts.
                if price < stop or i - entry_bar + 1 >= params.max_hold:
                    position = 0
            elif confirmed[i]:
                position = int(confirmed[i])
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapMemoryParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"close": [100., 100., 100., 104., 101., 101., 101.]}, index=index)
    for direction in (1, -1):
        derived = pd.DataFrame(
            {
                "beta": 1.0,
                "score": [direction, direction, 0., 0., 0., 0., 0.],
                "retained_gap": float(direction),
                "atr": 1.0,
            },
            index=index,
        )
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0, 0]
        assert result["size"].gt(0.0).all()
    derived["score"] = [1., 1., 0., 0., 0., 0., 0.]
    derived["retained_gap"] = 1.0
    params.max_hold = 2
    flat_data = data.assign(close=100.0)
    result = strategy.generate_signals(flat_data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    params.max_hold = 10
    ratchet_data = data.assign(close=[100., 100., 100., 104., 101., 101., 101.])
    derived["atr"] = [1., 1., 1., 1., 10., 10., 10.]
    result = strategy.generate_signals(ratchet_data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
