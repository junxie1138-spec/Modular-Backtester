from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 20
    block_window: int = 5
    strength_threshold: float = 0.65
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791316133"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 1 <= params.block_window < params.trend_window:
            raise ValueError("Require 1 <= block_window < trend_window")
        if params.atr_window < 1 or params.holding_bars < 1:
            raise ValueError("ATR and holding windows must be positive")
        if not np.isfinite(params.strength_threshold) or params.strength_threshold <= 0:
            raise ValueError("Strength threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Stop multiplier must be finite and positive")
        return int(max(params.trend_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["block_return"] = out["return"].rolling(params.block_window).sum()
        out["total_return"] = out["return"].rolling(params.trend_window).sum()
        # Every candidate block is entirely inside the current trend window.
        blocks = params.trend_window - params.block_window + 1
        out["best_block"] = out["block_return"].rolling(blocks).max()
        out["return_energy"] = np.sqrt(out["return"].pow(2).rolling(params.trend_window).sum())
        out["residual_strength"] = (
            (out["total_return"] - out["best_block"])
            / out["return_energy"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["eligible"] = (
            out["residual_strength"].ge(params.strength_threshold)
            & out["total_return"].gt(0.0)
            & out["best_block"].gt(0.0)
            & out["atr"].gt(0.0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        age = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # Today's open is the entry fill; ATR was known at yesterday's decision.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                pending = False
                age = 0
                if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(stop):
                    active = False
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.holding_bars:
                    active = False
            elif confirmed[i] and i + 1 < len(data):
                active = True
                pending = True
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    opens = np.array([100, 100, 100, 110, 107, 105, 110.0])
    closes = np.array([100, 100, 100, 107, 105, 110, 110.0])
    data = pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + 1,
        "low": np.minimum(opens, closes) - 1,
        "close": closes,
        "volume": 1000000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [2, 2, 2, 100, 100, 100, 100.0],
        "confirmed": [False, False, True, True, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    timed = strategy.generate_signals(
        data, derived, None, Params(holding_bars=3, stop_atr=100.0)
    ).data
    assert timed["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0]
    rising = pd.DataFrame(index=pd.date_range("2020-01-01", periods=40))
    rising["close"] = 100.0 * np.power(1.002, np.arange(40))
    rising["open"] = rising["close"]
    rising["high"] = rising["close"] + 1.0
    rising["low"] = rising["close"] - 1.0
    rising["volume"] = 1000000.0
    derived = strategy.indicators(rising, Params())
    assert not derived["confirmed"].iloc[:21].any()
    assert derived["confirmed"].iloc[21]
    assert np.isclose(derived["residual_strength"].iloc[-1], 15 / np.sqrt(20))
