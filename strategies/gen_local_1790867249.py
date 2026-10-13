from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NestedRangeParams:
    squeeze_bars: int = 3
    atr_period: int = 10
    trail_atr: float = 0.7
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[NestedRangeParams]):
    strategy_id = "gen_local_1790867249"

    @classmethod
    def params_type(cls):
        return NestedRangeParams

    def warmup_bars(self, params: NestedRangeParams) -> int:
        return int(max(params.squeeze_bars + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: NestedRangeParams) -> pd.DataFrame:
        if params.squeeze_bars < 1 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        out = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        inside = (high < high.shift(1)) & (low > low.shift(1))
        groups = (~inside).cumsum()
        out["inside_streak"] = inside.astype(int).groupby(groups).cumsum()
        window = params.squeeze_bars + 1
        out["ceiling"] = high.shift(1).rolling(window, min_periods=window).max()
        out["floor"] = low.shift(1).rolling(window, min_periods=window).min()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        ready = out["inside_streak"].shift(1).ge(params.squeeze_bars)
        out["first_up"] = ready & close.gt(out["ceiling"])
        out["first_down"] = ready & close.lt(out["floor"])
        # The second close must clear the same pre-breakout boundary.
        out["confirmed_up"] = (
            out["first_up"].shift(1, fill_value=False)
            & close.gt(out["ceiling"].shift(1))
            & close.gt(previous_close)
        )
        out["confirmed_down"] = (
            out["first_down"].shift(1, fill_value=False)
            & close.lt(out["floor"].shift(1))
            & close.lt(previous_close)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: NestedRangeParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        up = indicators["confirmed_up"].fillna(False).to_numpy(dtype=bool)
        down = indicators["confirmed_down"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position, held = 0, 0
        extreme, stop = np.nan, np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position, held = 0, 0
                extreme, stop = np.nan, np.nan
                continue
            if position:
                # This is a close of a bar actually held after the one-bar lag.
                held += 1
                if position == 1:
                    extreme = close if np.isnan(extreme) else max(extreme, close)
                    candidate = extreme - params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close <= stop
                else:
                    extreme = close if np.isnan(extreme) else min(extreme, close)
                    candidate = extreme + params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close >= stop
                if stopped or held >= params.hold_bars:
                    position, held = 0, 0
                    extreme, stop = np.nan, np.nan
            elif up[i] or down[i]:
                position = 1 if up[i] else -1
                held = 0
                extreme, stop = np.nan, np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {
            "open": [100, 100, 100, 106, 112, 114, 110, 109, 108],
            "high": [110, 108, 106, 113, 115, 116, 112, 111, 110],
            "low": [90, 92, 94, 105, 110, 111, 108, 107, 106],
            "close": [100, 100, 100, 112, 114, 115, 109, 108, 107],
            "volume": [1000] * 9,
        },
        index=index,
    )
    strategy = GeneratedStrategy()
    params = NestedRangeParams(squeeze_bars=2, atr_period=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    expected = [0, 0, 0, 0, 0, 1, 1, 0, 0]
    for direction in (1, -1):
        sample = data.copy()
        if direction == -1:
            sample["open"] = 200 - data["open"]
            sample["close"] = 200 - data["close"]
            sample["high"] = 200 - data["low"]
            sample["low"] = 200 - data["high"]
        result = strategy.generate_signals(
            sample, strategy.indicators(sample, params), ctx, params
        ).data
        assert result["signal"].tolist() == [direction * x for x in expected]
        assert (result["size"] > 0).all()
        prefix = sample.iloc[:6]
        prefix_result = strategy.generate_signals(
            prefix, strategy.indicators(prefix, params), ctx, params
        ).data
        assert prefix_result["signal"].equals(result["signal"].iloc[:6])


if __name__ == "__main__":
    _self_check()
