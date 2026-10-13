from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    season_window: int = 252
    min_samples: int = 16
    entry_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local_1791232705"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    def warmup_bars(self, params):
        return int(params.season_window + 4)

    def indicators(self, data, params):
        if params.season_window < 5 or params.min_samples < 1:
            raise ValueError("Lookback must be >= 5 and min_samples >= 1.")
        if not 0.0 < params.entry_threshold < 1.0:
            raise ValueError("entry_threshold must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = opening / closing.shift(1) - 1.0
        out["body"] = closing / opening - 1.0
        # Antisymmetric level-two signature of two (gap, body) increments.
        out["cross_forward"] = out["gap"].shift(1) * out["body"]
        out["cross_reverse"] = out["body"].shift(1) * out["gap"]
        out["signature_area"] = 0.5 * (out["cross_forward"] - out["cross_reverse"])
        out["area_scale"] = 0.5 * (out["cross_forward"].abs() + out["cross_reverse"].abs())
        out["area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["area"] = out["area"].where(out["area_scale"] != 0, 0.0)
        out["weekday"] = data.index.dayofweek
        # At t, the latest completed open-to-open trade was decided at t-2.
        out["training_area"] = out["area"].shift(2)
        out["training_weekday"] = out["weekday"].shift(2)
        out["realized_return"] = opening / opening.shift(1) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out["seasonal_edge"] = np.nan
        valid = out["training_area"].notna() & out["realized_return"].notna()
        for day in range(5):
            mask = valid & out["training_weekday"].eq(day)
            count_name = f"weekday_{day}_count"
            payoff_name = f"weekday_{day}_payoff"
            risk_name = f"weekday_{day}_risk"
            out[count_name] = mask.astype(float).rolling(params.season_window, min_periods=params.season_window).sum()
            out[payoff_name] = (out["training_area"] * out["realized_return"]).where(mask, 0.0).rolling(params.season_window, min_periods=params.season_window).sum()
            out[risk_name] = out["realized_return"].abs().where(mask, 0.0).rolling(params.season_window, min_periods=params.season_window).sum()
            edge = out[payoff_name] / out[risk_name].where(out[risk_name] > 0)
            eligible = out["weekday"].eq(day) & out[count_name].ge(params.min_samples)
            out.loc[eligible, "seasonal_edge"] = edge.loc[eligible]
        out["score"] = out["area"] * out["seasonal_edge"]
        out.loc[np.arange(len(out)) < self.warmup_bars(params), "score"] = np.nan
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        scores = indicators["score"].to_numpy(dtype=float)
        candidates = np.where(scores > params.entry_threshold, 1,
                              np.where(scores < -params.entry_threshold, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SignatureSeasonParams(season_window=30, min_samples=3)
    index = pd.bdate_range("2020-01-01", periods=120)
    t = np.arange(len(index), dtype=float)
    opening = 100.0 * np.exp(0.001 * t + 0.015 * np.sin(t))
    closing = opening * np.exp(0.01 * np.cos(1.7 * t))
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, closing) * 1.01,
                         "low": np.minimum(opening, closing) * 0.99,
                         "close": closing, "volume": 1000000.0}, index=index)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    features = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(features.iloc[:80], strategy.indicators(data.iloc[:80], params))
    assert features["score"].iloc[:strategy.warmup_bars(params)].isna().all()
    scores = pd.DataFrame({"score": [0.5, 0.0, -0.5, -0.5, np.nan, 0.5, 0.5]}, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], scores, context, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    mirrored = strategy.generate_signals(data.iloc[:7], -scores, context, params).data
    assert mirrored["signal"].tolist() == (-result["signal"]).tolist()
    assert result["size"].gt(0).all()
