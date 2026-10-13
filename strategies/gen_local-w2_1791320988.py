from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    signature_window: int = 12
    breakout_window: int = 40
    atr_window: int = 14
    min_area: float = 0.06
    trail_atr: float = 3.0
    holding_bars: int = 20

    def __post_init__(self):
        if min(self.signature_window, self.breakout_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if self.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        if not np.isfinite(self.min_area) or not 0 < self.min_area < 0.5:
            raise ValueError("min_area must lie between zero and 0.5.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791320988"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.signature_window, params.breakout_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        n = params.signature_window
        # Antisymmetric second-level signature of normalized time and log price.
        # Later gains contribute positively; earlier gains contribute negatively.
        area = pd.Series(0.0, index=data.index)
        for lag in range(n):
            area = area + ((n - 1 - 2 * lag) / (2.0 * n)) * out["log_return"].shift(lag)
        out["signature_area"] = area
        out["path_variation"] = out["log_return"].abs().rolling(n, min_periods=n).sum()
        out["area_score"] = area / out["path_variation"].replace(0.0, np.nan)
        out["prior_high"] = data["high"].rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max().shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        area = indicators["area_score"].to_numpy(dtype=float)
        prior_high = indicators["prior_high"].to_numpy(dtype=float)
        valid = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(atr) & (atr > 0)
            & np.isfinite(area) & np.isfinite(prior_high)
        )
        entry = valid & (close > prior_high) & (area > params.min_area)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        allocation = 0.25
        for i in range(len(data)):
            if active:
                # Entry was decided yesterday; today's close is the first held close.
                age += 1
                if np.isfinite(close[i]) and close[i] > 0:
                    high_water = max(high_water, close[i])
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, high_water - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    stopped = False
                if stopped or age >= params.holding_bars:
                    active = False
            elif entry[i]:
                active = True
                high_water = -np.inf
                stop = -np.inf
                age = 0
                allocation = float(0.25 + 0.70 * np.clip(
                    area[i] / (2.0 * params.min_area), 0.0, 1.0
                ))
            if active:
                signals[i] = 1
                sizes[i] = allocation
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    close = np.array([100., 101., 106., 104., 102., 103., 104., 105.])
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.},
        index=index,
    )
    indicators = pd.DataFrame(
        {"atr": [1., 1., 1., 10., 10., 10., 10., 10.],
         "area_score": [0.09] * 8, "prior_high": [99.] + [200.] * 7},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.775)
    assert result["size"].gt(0).all()
    timed = strategy.generate_signals(
        data, indicators, None, Params(holding_bars=2)
    ).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    warm = strategy.indicators(data, Params())
    assert strategy.generate_signals(data, warm, None, Params()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
