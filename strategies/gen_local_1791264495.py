from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 12
    percentile_window: int = 126
    entry_quantile: float = 0.15
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791264495"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(15, params.complexity_window + params.percentile_window + 6)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.complexity_window < 3 or params.percentile_window < 5
                or not 0.0 < params.entry_quantile < 0.5
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        triples = pd.concat([close.shift(2), close.shift(1), close], axis=1)
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        valid = np.isfinite(triples.to_numpy()).all(axis=1)
        out["pattern"] = pd.Series(
            9 * order[:, 0] + 3 * order[:, 1] + order[:, 2],
            index=data.index, dtype=float
        ).where(valid)
        for code in (5, 7, 11, 15, 19, 21):
            out[f"prob_{code}"] = (
                out["pattern"].eq(code).astype(float).where(valid)
                .rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            )
        probabilities = out[[f"prob_{code}" for code in (5, 7, 11, 15, 19, 21)]]
        out["complexity"] = -(
            probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out["collapse"] = out["complexity"].diff(3)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["anchor"] = close.rolling(10, min_periods=10).mean()
        out["stretch"] = (close - out["anchor"]) / out["atr"].where(out["atr"] > 0)
        for name in ("collapse", "stretch"):
            out[f"{name}_cutoff"] = (
                out[name].rolling(params.percentile_window, min_periods=params.percentile_window)
                .quantile(params.entry_quantile).shift(1)
            )
        out["entry"] = (
            (out["collapse"] < out["collapse_cutoff"])
            & (out["collapse"] < 0)
            & (out["stretch"] < out["stretch_cutoff"])
            & (out["stretch"] < 0)
            & (close < previous) & (previous < close.shift(2))
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                age = 0
                high_water = stop = -np.inf
                continue
            if active:
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= params.hold_bars:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                age = 0
                # Entry fills next bar; its close initializes the in-trade high.
                high_water = stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 105., 102., 101., 100., 99.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 5., 5., 5., 5.],
        "entry": [True, False, False, False, False, False, False]
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.
    indicators["atr"] = 1.
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    index = pd.date_range("2020-01-01", periods=200)
    flat = pd.DataFrame({
        "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.
    }, index=index)
    derived = strategy.indicators(flat, params)
    assert np.isfinite(derived["collapse_cutoff"].iloc[strategy.warmup_bars(params)])
    assert not derived["entry"].any()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
