from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 10
    horizon_bars: int = 8
    history_bars: int = 504
    neighbors: int = 20
    threshold: float = 0.20


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791256387"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return params.history_bars + params.path_bars + params.horizon_bars + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.path_bars, params.horizon_bars, params.history_bars, params.neighbors) < 1:
            raise ValueError("Lookbacks and neighbors must be positive.")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive.")
        if params.neighbors > params.history_bars - params.path_bars + 1:
            raise ValueError("Insufficient history for neighbors.")
        log_close = np.log(data["close"].where(data["close"] > 0))
        returns = log_close.diff().replace([np.inf, -np.inf], np.nan)
        energy = returns.pow(2).rolling(params.path_bars, min_periods=params.path_bars).sum()
        scale = np.sqrt(energy).replace(0.0, np.nan)
        variance_prefix = pd.Series(0.0, index=data.index)
        price_prefix = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        for lag in range(params.path_bars - 1, -1, -1):
            increment = returns.shift(lag)
            variance = increment.pow(2)
            area = area + 0.5 * (variance_prefix * increment - price_prefix * variance)
            variance_prefix = variance_prefix + variance
            price_prefix = price_prefix + increment
        out = pd.DataFrame(index=data.index)
        out["log_close"] = log_close
        out["return"] = returns
        out["energy"] = energy
        out["scale"] = scale
        out["area"] = area / (energy * scale)
        out["displacement"] = price_prefix / scale
        out["mature_area"] = out["area"].shift(params.horizon_bars)
        out["mature_displacement"] = out["displacement"].shift(params.horizon_bars)
        out["mature_outcome"] = log_close.diff(params.horizon_bars) / scale.shift(params.horizon_bars)
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["area", "displacement"]].to_numpy()
        archive = out[["mature_area", "mature_displacement"]].to_numpy()
        outcomes = out["mature_outcome"].to_numpy()
        forecast = np.full(len(data), np.nan)
        # ponytail: O(bars * history); use a neighbor index if intraday histories become large.
        for t in range(GeneratedStrategy.warmup_bars(params), len(data)):
            if not np.isfinite(features[t]).all():
                continue
            start = max(0, t - params.history_bars + 1)
            stop = t - params.path_bars + 1
            candidates = archive[start:stop]
            labels = outcomes[start:stop]
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(labels)
            candidates, labels = candidates[valid], labels[valid]
            if len(labels) < params.neighbors:
                continue
            direct = ((candidates - features[t]) ** 2 * np.array([1.0, 0.25])).sum(axis=1)
            reflected = ((-candidates - features[t]) ** 2 * np.array([1.0, 0.25])).sum(axis=1)
            orientation = np.sign(reflected - direct)
            distance = np.minimum(direct, reflected)
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            forecast[t] = np.mean(labels[nearest] * orientation[nearest])
        out["forecast"] = forecast
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        forecast = indicators["forecast"].to_numpy()
        entry = np.where(np.isfinite(forecast) & (forecast >= params.threshold), 1,
                         np.where(np.isfinite(forecast) & (forecast <= -params.threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for t, direction in enumerate(entry):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[t] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=6))
    indicators = pd.DataFrame({"forecast": [1.0, -1.0, -1.0, 1.0, 1.0, np.nan]}, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, -1, 0, 1]
    mirrored = GeneratedStrategy.generate_signals(None, data, -indicators, None, params).data
    assert mirrored["signal"].equals(-result["signal"])
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
