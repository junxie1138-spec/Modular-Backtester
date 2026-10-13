from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalBreakoutParams:
    fractional_order: float = 0.4
    fractional_window: int = 64
    breakout_window: int = 30
    breakout_buffer: float = 0.15
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[FractionalBreakoutParams]):
    strategy_id = "gen_local-w2_1791253594"

    @classmethod
    def params_type(cls):
        return FractionalBreakoutParams

    def warmup_bars(self, params: FractionalBreakoutParams) -> int:
        return int(params.fractional_window + params.breakout_window)

    def indicators(self, data: pd.DataFrame, params: FractionalBreakoutParams) -> pd.DataFrame:
        if not (
            0.0 < params.fractional_order < 1.0
            and params.fractional_window >= 2
            and params.breakout_window >= 2
            and np.isfinite(params.breakout_buffer)
            and params.breakout_buffer >= 0.0
            and params.holding_bars >= 1
        ):
            raise ValueError("Invalid fractional breakout parameters")
        close = data["close"].astype(float).where(
            np.isfinite(data["close"]) & (data["close"] > 0.0)
        )
        fractional_close = close.copy()
        weight = 1.0
        # Fixed-width fractional differencing: lag k has coefficient (-1)^k binom(d, k).
        for k in range(1, params.fractional_window):
            weight *= (k - 1.0 - params.fractional_order) / k
            fractional_close = fractional_close + weight * close.shift(k)
        prior = fractional_close.shift(1).rolling(params.breakout_window)
        upper = prior.max()
        lower = prior.min()
        noise = prior.std(ddof=0).where(lambda x: x > 0.0)
        width = upper - lower
        inner = fractional_close.between(lower + 0.25 * width, upper - 0.25 * width)
        long_excess = (fractional_close - upper) / noise - params.breakout_buffer
        short_excess = (lower - fractional_close) / noise - params.breakout_buffer
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[long_excess.gt(0.0)] = 1
        direction.loc[short_excess.gt(0.0)] = -1
        strength = pd.concat([long_excess, short_excess], axis=1).max(axis=1)
        strength = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "upper": upper,
            "lower": lower,
            "noise": noise,
            "inner": inner.fillna(False),
            "long_excess": long_excess,
            "short_excess": short_excess,
            "direction": direction,
            "strength": strength,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalBreakoutParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        inner = indicators["inner"].fillna(False).to_numpy(dtype=bool)
        armed = False
        position = 0
        exit_bar = -1
        entry_size = 0.25
        for i in range(n):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    armed = False
                    continue
            elif armed and directions[i] != 0:
                position = int(directions[i])
                exit_bar = i + params.holding_bars
                entry_size = float(0.25 + 0.70 * np.clip(strengths[i] / 1.5, 0.0, 1.0))
                armed = False
            elif inner[i]:
                armed = True
            signals[i] = position
            if position != 0:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = FractionalBreakoutParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=180)
    close = 100.0 + np.arange(180) * 0.1 + np.sin(np.arange(180))
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    assert indicators["upper"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    changed = data.copy()
    changed.loc[index[-1], "close"] *= 2.0
    pd.testing.assert_frame_equal(
        indicators.iloc[:-1], strategy.indicators(changed, params).iloc[:-1]
    )
    forced = pd.DataFrame({
        "direction": 0, "strength": 0.0, "inner": True,
    }, index=index)
    forced.loc[index[1], ["direction", "strength", "inner"]] = [1, 0.3, False]
    forced.loc[index[5], ["direction", "strength", "inner"]] = [-1, 1.5, False]
    forced.loc[index[22], ["direction", "strength", "inner"]] = [-1, 1.5, False]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=94)
    output = strategy.generate_signals(data, forced, ctx, params).data
    expected = np.zeros(180, dtype=int)
    expected[2:20] = 1
    expected[23:41] = -1
    assert np.array_equal(output["signal"].to_numpy(), expected)
    assert np.allclose(output["size"].iloc[2:20], 0.39)
    assert np.allclose(output["size"].iloc[23:41], 0.95)
    assert output["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
