from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    fractional_order: float = 0.4
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1790905435"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        # 64-tap filter + 20 prior observations + previous position.
        return 85

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if not np.isfinite(params.fractional_order) or not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be positive")
        close = data["close"].astype(float)
        weights = np.ones(64, dtype=float)
        for lag in range(1, 64):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # Fixed-width fractional difference: sum w[j] * close[t-j].
        fractional = close * weights[0]
        for lag in range(1, 64):
            fractional = fractional + close.shift(lag) * weights[lag]
        lower = fractional.rolling(20, min_periods=20).min().shift(1)
        upper = fractional.rolling(20, min_periods=20).max().shift(1)
        width = (upper - lower).where(upper > lower)
        position = (fractional - lower) / width
        previous = close.shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous).abs(),
                (data["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        entry = (
            (position.shift(1) <= 0.2)
            & (position > 0.2)
            & (position < 0.45)
            & (close > previous)
            & (atr > 0.0)
        ).fillna(False)
        return pd.DataFrame(
            {
                "fractional_close": fractional,
                "lower": lower,
                "upper": upper,
                "relative_position": position,
                "true_range": true_range,
                "atr": atr,
                "entry": entry,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry_price = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                # The prior close's decision is filled at this session's open.
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    raise ValueError("entry open must be finite and positive")
                stop = entry_price - 2.0 * atrs[i - 1]
                peak = entry_price
                active, pending, armed, age = True, False, False, 0
            if active:
                age += 1
                valid = np.isfinite(lows[i]) and np.isfinite(closes[i]) and np.isfinite(atrs[i])
                # Test the previously established stop before updating it.
                # Stop breaches generate next-open exits, not intrabar fills.
                if not valid or lows[i] <= stop or age >= 4:
                    active = False
                else:
                    peak = max(peak, closes[i])
                    armed = armed or closes[i] >= entry_price * 1.01
                    if armed:
                        stop = max(stop, entry_price, peak - params.trail_atr * atrs[i])
                    raw[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {
            "open": [100.0] * 7,
            "high": [100.5, 101.5, 102.5, 101.5, 100.5, 100.5, 100.5],
            "low": [99.5, 99.5, 100.2, 100.2, 99.5, 99.5, 99.5],
            "close": [100.0, 101.2, 102.0, 101.0, 100.0, 100.0, 100.0],
            "volume": [1000000.0] * 7,
        },
        index=index,
    )
    features = pd.DataFrame(
        {"atr": [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
         "entry": [True, False, False, False, False, False, False]},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = ElasticParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=85)
    result = strategy.generate_signals(data, features, ctx, params).data
    # A larger ATR must not lower the 100.5 stop; its breach exits next open.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    features["atr"] = 1.0
    data["low"] = 100.8
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
