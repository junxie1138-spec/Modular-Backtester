from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 18
    percentile_window: int = 126
    position_window: int = 20
    atr_window: int = 14
    threshold_quantile: float = 0.8
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790964057"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.complexity_window + params.percentile_window + 3,
            params.position_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.complexity_window, params.percentile_window,
               params.position_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not 0.5 < params.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie between 0.5 and 1.")
        if params.trail_atr <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Positive trail_atr and a 3-5 bar hold are required.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        triples = pd.concat([close.shift(2), close.shift(1), close], axis=1)
        valid = np.isfinite(triples.to_numpy()).all(axis=1)
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out["ordinal_code"] = pd.Series(codes, index=data.index).where(valid)
        permutations = (5, 7, 11, 15, 19, 21)
        for code in permutations:
            occurrences = out["ordinal_code"].eq(code).astype(float).where(valid)
            out[f"p_{code}"] = occurrences.rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
        probabilities = out[[f"p_{code}" for code in permutations]]
        entropy_terms = probabilities * np.log(probabilities.clip(lower=1e-12))
        out["entropy"] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = (
            (probabilities - 1.0 / 6.0).pow(2).sum(axis=1, min_count=6) * 6.0 / 5.0
        )
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["complexity_change"] = out["complexity"].diff()
        out["complexity_threshold"] = out["complexity"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.threshold_quantile)
        out["close_high"] = close.rolling(params.position_window).max()
        out["close_low"] = close.rolling(params.position_window).min()
        span = (out["close_high"] - out["close_low"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["close_low"]) / span
        position_history = out["relative_position"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["position_upper"] = position_history.quantile(params.threshold_quantile)
        out["position_lower"] = position_history.quantile(1.0 - params.threshold_quantile)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = np.isfinite(out[[
            "complexity", "complexity_change", "complexity_threshold",
            "relative_position", "position_upper", "position_lower", "atr"
        ]]).all(axis=1) & out["atr"].gt(0)
        structured = (
            ready & out["complexity"].gt(out["complexity_threshold"])
            & out["complexity_change"].gt(0)
        )
        out["entry"] = np.select([
            structured & out["relative_position"].gt(out["position_upper"]),
            structured & out["relative_position"].lt(out["position_lower"]),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, stop = -np.inf, -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position, age = 0, 0
                high_water, stop = -np.inf, -np.inf
                continue
            if position:
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                # The required downward high-water stop also takes profit on shorts.
                if close < stop or age >= params.hold_bars:
                    position, age = 0, 0
                    high_water, stop = -np.inf, -np.inf
            elif entries[i]:
                position, age = int(entries[i]), 0
                # Seed on the decision close; subsequent closes include the fill bar.
                high_water = close
                stop = close - params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 104., 103., 102., 101., 100.]}, index=index)
    ind = pd.DataFrame({"atr": [1., 1., 10., 10., 10., 10.],
                        "entry": [1, 0, 0, 0, 0, 0]}, index=index)
    ctx = StrategyContext("SPY", "1d", 0)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    ind["entry"] = [-1, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, 0, 0]
    data["close"], ind["atr"] = 100., 1.
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, -1, 0]
