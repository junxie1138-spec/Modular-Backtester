from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    volatility_window: int = 12
    shape_step: int = 4
    memory: int = 252
    neighbors: int = 16
    outcome_bars: int = 8
    smoothing: int = 5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791343424"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(params.memory + params.volatility_window + 1
                   + 2 * params.shape_step + params.outcome_bars
                   + params.smoothing)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if min(params.volatility_window, params.shape_step, params.memory,
               params.neighbors, params.outcome_bars, params.smoothing) < 1:
            raise ValueError("Lookbacks and neighbor count must be positive.")
        if params.neighbors > params.memory:
            raise ValueError("Neighbor count must not exceed memory.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap_return"] = np.log(opening / close.shift(1))
        out["session_return"] = np.log(close / opening)
        out["gap_vol"] = out["gap_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["session_vol"] = out["session_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["total_vol"] = np.sqrt(out["gap_vol"] ** 2 + out["session_vol"] ** 2)
        variance = out["total_vol"].pow(2).replace(0.0, np.nan)
        out["allocation"] = out["gap_vol"].pow(2) / variance
        out["allocation_lag1"] = out["allocation"].shift(params.shape_step)
        out["allocation_lag2"] = out["allocation"].shift(2 * params.shape_step)
        safe_vol = out["total_vol"].replace(0.0, np.nan)
        out["vol_shape1"] = np.log(safe_vol / safe_vol.shift(params.shape_step))
        out["vol_shape2"] = np.log(safe_vol / safe_vol.shift(2 * params.shape_step))
        out["realized_return"] = np.log(close / close.shift(params.outcome_bars))
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["allocation", "allocation_lag1", "allocation_lag2",
                        "vol_shape1", "vol_shape2"]].to_numpy(dtype=float)
        vol = out["total_vol"].to_numpy(dtype=float)
        realized = out["realized_return"].to_numpy(dtype=float)
        score = np.full(len(data), np.nan)
        horizon = params.outcome_bars
        # ponytail: O(bars * memory) search; use an index if history becomes large.
        for t in range(len(data)):
            if not np.isfinite(features[t]).all():
                continue
            last = t - horizon
            if last < 0:
                continue
            candidates = np.arange(max(0, last - params.memory + 1), last + 1)
            # Every outcome ends on or before the current decision bar.
            labels = realized[candidates + horizon] / (
                np.where(vol[candidates] > 0, vol[candidates], np.nan)
                * np.sqrt(horizon)
            )
            valid = (np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(labels) & (vol[candidates] > 0))
            candidates, labels = candidates[valid], labels[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.square(features[candidates] - features[t]).sum(axis=1)
            nearest = np.argsort(distances, kind="stable")[:params.neighbors]
            score[t] = labels[nearest].mean()
        out["analog_score"] = score
        out["smoothed_score"] = out["analog_score"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same condition opens and maintains the position; its reversal exits.
        df["signal"] = (indicators["smoothed_score"].fillna(0.0) > 0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 420)))
    opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, 420))
    sample = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) * 1.002,
        "low": np.minimum(opening, close) * 0.998, "close": close,
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=420, freq="B"))
    params = AnalogParams()
    full = GeneratedStrategy.indicators(sample, params)
    prefix = GeneratedStrategy.indicators(sample.iloc[:360], params)
    np.testing.assert_allclose(full.iloc[:360].to_numpy(), prefix.to_numpy(),
                               equal_nan=True)
    signals = GeneratedStrategy.generate_signals(sample, full, None, params).data
    expected = (full["smoothed_score"].fillna(0.0) > 0.0).astype(int)
    expected = expected.shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected.rename("signal"))
    assert full["analog_score"].notna().any()
    assert set(signals["signal"].unique()) <= {0, 1}
    assert (signals["size"] > 0).all()
