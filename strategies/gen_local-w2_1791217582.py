from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 24
    min_entropy: float = 0.90
    breakout_window: int = 20
    atr_window: int = 14
    atr_multiple: float = 3.0
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791217582"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.complexity_window + 4,
                   params.breakout_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.complexity_window, params.breakout_window,
               params.atr_window, params.max_hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not 0.0 <= params.min_entropy <= 1.0 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("Invalid entropy threshold or ATR multiple.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = (4 * a.gt(b).astype(int) +
                          2 * a.gt(c).astype(int) +
                          b.gt(c).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 3, 4, 6, 7):
            p = out["pattern"].eq(pattern).astype(float).rolling(
                params.complexity_window, min_periods=params.complexity_window).mean()
            out[f"pattern_probability_{pattern}"] = p
            entropy -= p * np.log(p.where(p > 0, 1.0))
        complete = valid.astype(int).rolling(
            params.complexity_window, min_periods=params.complexity_window).sum().eq(params.complexity_window)
        out["entropy"] = (entropy / np.log(6.0)).where(complete)
        out["prior_entropy"] = out["entropy"].shift(1)
        out["breakout_high"] = data["high"].rolling(
            params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = np.isfinite(close) & np.isfinite(atr) & (atr > 0)
        entry = (indicators["prior_entropy"].ge(params.min_entropy) &
                 data["close"].gt(indicators["breakout_high"])).fillna(False).to_numpy() & ready
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        held = 0
        for i in range(len(data)):
            if active:
                held += 1
                if not ready[i]:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.atr_multiple * atr[i])
                    if close[i] <= stop or held >= params.max_hold_bars:
                        active = False
            elif entry[i]:
                active = True
                # Entry fills next bar; only subsequent closes enter the high-water mark.
                high_water = -np.inf
                stop = -np.inf
                held = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 100., 110., 107., 106., 105., 104.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 10., 10., 10., 10.],
        "prior_entropy": [1., 0., 0., 0., 0., 0., 0.],
        "breakout_high": [99., 200., 200., 200., 200., 200., 200.]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, None, Params(max_hold_bars=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
