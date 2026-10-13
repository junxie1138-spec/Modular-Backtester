from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history_bars: int = 756
    conviction: float = 0.35

    def __post_init__(self):
        if self.history_bars < 252 or not 0.0 < self.conviction <= 1.0:
            raise ValueError("history_bars must be >= 252; conviction must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1790977319"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars) + 34

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = pd.to_numeric(data["close"], errors="coerce")
        returns = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame({"log_return": returns}, index=data.index)
        shape = pd.concat([returns.shift(lag) for lag in range(15)], axis=1)
        out["shape_norm"] = np.sqrt(shape.pow(2).sum(axis=1, skipna=False))
        shape = shape.div(out["shape_norm"].where(out["shape_norm"] > 0), axis=0)
        for lag in range(15):
            out[f"shape_{lag}"] = shape.iloc[:, lag]
        out["trailing_return_18"] = returns.rolling(18, min_periods=18).sum()
        out["analog_forecast"] = np.nan
        out["analog_score"] = np.nan
        out["analog_count"] = 0
        vectors = shape.to_numpy(dtype=float)
        outcomes = out["trailing_return_18"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        score = np.full(len(data), np.nan)
        count = np.zeros(len(data), dtype=int)
        # Fixed 33-bar blocks keep analog patterns and outcomes mutually disjoint.
        endpoints = np.arange(15, len(data), 33)
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(vectors[t]).all():
                continue
            candidates = endpoints[(endpoints >= t - params.history_bars) & (endpoints <= t - 33)]
            valid = np.isfinite(vectors[candidates]).all(axis=1) & np.isfinite(outcomes[candidates + 18])
            candidates = candidates[valid]
            if len(candidates) < 7:
                continue
            distance = np.sum((vectors[candidates] - vectors[t]) ** 2, axis=1)
            nearest = candidates[np.argsort(distance, kind="stable")[:7]]
            realized = outcomes[nearest + 18]
            forecast[t] = realized.mean()
            rms = np.sqrt(np.mean(realized ** 2))
            score[t] = np.clip(forecast[t] / rms, -1.0, 1.0) if rms > 0 else 0.0
            count[t] = len(nearest)
        out["analog_forecast"] = forecast
        out["analog_score"] = score
        out["analog_count"] = count
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        score = indicators["analog_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        desired = np.where(score >= params.conviction, 1, np.where(score <= -params.conviction, -1, 0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for t, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signal[t] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams(history_bars=252)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:400], params)
    assert np.allclose(full.iloc[:400].to_numpy(), prefix.to_numpy(), equal_nan=True)
    result = strategy.generate_signals(data, full, ctx, params).data
    assert result["signal"].isin([-1, 0, 1]).all() and (result["size"] > 0).all()
    assert (result["signal"].iloc[:strategy.warmup_bars(params)] == 0).all()
    sample = pd.DataFrame({"analog_score": [0.8, 0.0, -0.8, -0.8, 0.0, 0.8, 0.8]}, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], sample, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_indicators = strategy.indicators(flat, params)
    assert (strategy.generate_signals(flat, flat_indicators, ctx, params).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
