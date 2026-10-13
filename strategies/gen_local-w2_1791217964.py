from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalityParams:
    history: int = 126
    edge: float = 0.12
    trail_atr: float = 1.5
    atr_period: int = 14
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalityParams]):
    strategy_id = "gen_local-w2_1791217964"

    @classmethod
    def params_type(cls):
        return SeasonalityParams

    def warmup_bars(self, params: SeasonalityParams) -> int:
        return max(params.history, params.atr_period) + 2

    def indicators(self, data: pd.DataFrame, params: SeasonalityParams) -> pd.DataFrame:
        if params.history < 60 or params.atr_period < 2:
            raise ValueError("history must be >= 60 and atr_period >= 2")
        if not 3 <= params.hold_days <= 5 or params.edge <= 0 or params.trail_atr <= 0:
            raise ValueError("hold_days must be 3-5; edge and trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["return"] = ret
        out["weekday"] = weekday
        out["mean_return"] = ret.rolling(params.history, min_periods=params.history).mean()
        out["volatility"] = ret.rolling(params.history, min_periods=params.history).std()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        for day in range(5):
            masked = ret.where(weekday.eq(day))
            count = masked.rolling(params.history, min_periods=1).count()
            total = masked.fillna(0.0).rolling(
                params.history, min_periods=params.history
            ).sum()
            out[f"weekday_mean_{day}"] = (total / count.replace(0, np.nan)).where(count >= 8)
        forecast = pd.Series(0.0, index=data.index)
        # ponytail: weekday template ignores holidays; use an exchange calendar if material.
        for offset in range(1, params.hold_days + 1):
            target = (weekday + offset) % 5
            selected = pd.Series(np.nan, index=data.index)
            for day in range(5):
                selected = selected.where(target.ne(day), out[f"weekday_mean_{day}"])
            forecast = forecast + selected - out["mean_return"]
        out["seasonal_excess"] = forecast
        out["score"] = forecast / (
            out["volatility"].replace(0.0, np.nan) * np.sqrt(params.hold_days)
        )
        score = out["score"].replace([np.inf, -np.inf], np.nan)
        out["candidate"] = pd.Series(
            np.select([score.gt(params.edge), score.lt(-params.edge)], [1, -1], default=0),
            index=data.index, dtype=int
        )
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SeasonalityParams
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        peak = floor = -np.inf
        trough = ceiling = np.inf
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if valid:
                    peak = max(peak, price)
                    floor = max(floor, peak - params.trail_atr * atr)
                    trough = min(trough, price)
                    ceiling = min(ceiling, trough + params.trail_atr * atr)
                # Both directions obey the required highest-close drawdown exit.
                # Shorts additionally have a ratcheting adverse-price ceiling.
                stopped = valid and (price < floor or (position == -1 and price > ceiling))
                if not valid or stopped or age >= params.hold_days:
                    position = 0
                signals[i] = position
                continue
            if valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                # Initialize extrema on the first held bar, after the next-bar fill.
                peak = floor = -np.inf
                trough = ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 100., 100., 103., 100., 100.]}, index=index)
    features = pd.DataFrame({"atr": 1.0, "confirmed": [0, 1, 1, 1, 1, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, features, None, SeasonalityParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, features, None, SeasonalityParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1]
    features["confirmed"] = [0, -1, -1, -1, -1, 0]
    data["close"] = [100., 100., 100., 101., 102., 102.]
    result = strategy.generate_signals(data, features, None, SeasonalityParams()).data
    assert result["signal"].tolist() == [0, 0, -1, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
