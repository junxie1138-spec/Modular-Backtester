from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fraction: float = 0.45
    fractional_window: int = 40
    recent_window: int = 3
    reference_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.90
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791250120"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.fractional_window + params.recent_window
            + params.reference_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (
            0.0 < params.fraction < 1.0
            and params.fractional_window >= 2
            and params.recent_window >= 2
            and params.reference_window >= 2
            and params.percentile_window >= 2
            and 0.0 < params.entry_percentile < 1.0
            and params.atr_window >= 1
            and np.isfinite(params.trail_atr) and params.trail_atr > 0.0
            and params.max_hold >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        fractional = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; enlarge the window if tail sensitivity matters.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fraction) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["recent_level"] = fractional.rolling(params.recent_window).mean()
        out["reference_level"] = fractional.shift(params.recent_window).rolling(
            params.reference_window
        ).mean()
        out["reference_scale"] = fractional.shift(params.recent_window).rolling(
            params.reference_window
        ).std(ddof=0).replace(0.0, np.nan)
        out["change_score"] = (
            (out["recent_level"] - out["reference_level"]) / out["reference_scale"]
        ).replace([np.inf, -np.inf], np.nan)
        out["absolute_score"] = out["change_score"].abs()
        out["threshold"] = out["absolute_score"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_percentile)
        crossing = (
            out["absolute_score"].gt(out["threshold"])
            & out["absolute_score"].shift(1).le(out["threshold"].shift(1))
        )
        out["event"] = np.select(
            [crossing & out["change_score"].gt(0),
             crossing & out["change_score"].lt(0)],
            [1, -1], default=0
        ).astype(int)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = 0.0
        high_stop = -np.inf
        short_stop = np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        events = indicators["event"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0.0):
                position = 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                high_stop = max(high_stop, high_water - params.trail_atr * atr)
                short_stop = min(short_stop, low_water + params.trail_atr * atr)
                # High-close drawdown covers shorts too; the low-close trail caps adverse rebounds.
                if (
                    close <= high_stop
                    or (position == -1 and close >= short_stop)
                    or age >= params.max_hold
                ):
                    position = 0
            elif events[i] != 0:
                position = int(events[i])
                age = 0
                high_water = low_water = close
                high_stop = close - params.trail_atr * atr
                short_stop = close + params.trail_atr * atr
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 104., 110., 108., 106., 105.]}, index=index)
    derived = pd.DataFrame(
        {"atr": [1., 1., 1., 3., 3., 3.], "event": [1, 0, 0, 0, 0, 0]},
        index=index,
    )
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data["close"] = [100., 99., 98., 97., 96., 95.]
    derived["atr"] = 1.0
    derived["event"] = [-1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0]
