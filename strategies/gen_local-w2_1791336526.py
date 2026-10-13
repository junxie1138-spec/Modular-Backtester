from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ordinal_window: int = 12
    baseline_window: int = 12
    entropy_drop: float = 0.10
    centroid_window: int = 8
    drift_lag: int = 3
    min_drift_atr: float = 0.20
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791336526"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ordinal_window + params.baseline_window + 3,
                       params.centroid_window + params.drift_lag,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.div(close.shift(1)).sub(1).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack((out["return"].shift(2),
                                   out["return"].shift(1), out["return"]))
        valid = np.isfinite(triples).all(axis=1)
        # Stable ties rank the earlier observation first.
        order = np.argsort(triples, axis=1, kind="stable")
        codes = order @ np.array([1, 3, 9])
        out["ordinal_code"] = pd.Series(codes, index=data.index).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (21, 15, 19, 7, 11, 5):
            frequency = out["ordinal_code"].eq(code).astype(float).where(valid)
            probability = frequency.rolling(params.ordinal_window,
                                             min_periods=params.ordinal_window).mean()
            out[f"pattern_probability_{code}"] = probability
            entropy -= probability * np.log(probability.where(probability > 0, 1.0))
        out["entropy"] = entropy / np.log(6.0)
        out["entropy_baseline"] = out["entropy"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window).mean()
        previous_close = close.shift(1)
        true_range = pd.concat((data["high"] - data["low"],
                                (data["high"] - previous_close).abs(),
                                (data["low"] - previous_close).abs()), axis=1).max(axis=1)
        out["true_range"] = true_range.where(close.notna() & previous_close.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean().where(lambda x: x > 0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["volume_sum"] = volume.rolling(
            params.centroid_window, min_periods=params.centroid_window).sum()
        out["weighted_close_sum"] = (close * volume).rolling(
            params.centroid_window, min_periods=params.centroid_window).sum()
        out["centroid"] = out["weighted_close_sum"].div(out["volume_sum"].where(out["volume_sum"] > 0))
        out["centroid_drift_atr"] = out["centroid"].sub(
            out["centroid"].shift(params.drift_lag)).div(out["atr"])
        out["complexity_agrees"] = (
            (out["entropy_baseline"] >= 0.80)
            & (out["entropy"] <= 0.85)
            & (out["entropy_baseline"] - out["entropy"] >= params.entropy_drop))
        out["trend_agrees"] = out["centroid_drift_atr"] >= params.min_drift_atr
        out["entry"] = (out["complexity_agrees"] & out["trend_agrees"]
                        & out["atr"].notna() & close.notna()).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        peak = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            usable = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                if usable:
                    # The first held close is on the bar following the entry decision.
                    peak = max(peak, close)
                    stop = max(stop, peak - params.trail_atr * atr)
                if not usable or close <= stop or held >= params.max_hold:
                    active = False
            elif usable and entries[i]:
                active = True
                held = 0
                peak = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 102., 105., 103., 102., 100.]}, index=index)
    indicators = pd.DataFrame({"atr": [1., 1., 1., 10., 10., 10.],
                               "entry": [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
