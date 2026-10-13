from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    history_bars: int = 252
    edge_threshold: float = 0.002


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791331280"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        if params.history_bars < 32:
            raise ValueError("history_bars must be at least 32")
        if not np.isfinite(params.edge_threshold) or params.edge_threshold < 0:
            raise ValueError("edge_threshold must be finite and nonnegative")
        return int(params.history_bars) + 16

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        energy_columns = []
        for scale in (2, 4, 8):
            half = scale // 2
            out[f"block_{scale}"] = out["return"].rolling(half, min_periods=half).sum()
            out[f"haar_{scale}"] = (
                out[f"block_{scale}"] - out[f"block_{scale}"].shift(half)
            ) / np.sqrt(scale)
            name = f"energy_{scale}"
            out[name] = out[f"haar_{scale}"].pow(2).rolling(5, min_periods=5).mean()
            energy_columns.append(name)
        out["total_energy"] = out[energy_columns].sum(axis=1, min_count=3)
        feature_columns = []
        for scale in (2, 4, 8):
            name = f"share_{scale}"
            out[name] = out[f"energy_{scale}"] / out["total_energy"].where(out["total_energy"] > 0)
            feature_columns.append(name)
        out["drift_sum"] = out["return"].rolling(8, min_periods=8).sum()
        out["absolute_sum"] = out["return"].abs().rolling(8, min_periods=8).sum()
        out["orientation"] = out["drift_sum"] / out["absolute_sum"].where(out["absolute_sum"] > 0)
        feature_columns.append("orientation")
        # At maturity k, the shape belongs to k-3; its next-open two-day
        # outcome is log(open[k] / open[k-2]). No future labels are read.
        out["matured_outcome"] = np.log(data["open"].where(data["open"] > 0)).diff(2)
        features = out[feature_columns].to_numpy(dtype=float)
        matured_features = out[feature_columns].shift(3).to_numpy(dtype=float)
        outcomes = out["matured_outcome"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        weights = np.array([1.0, 1.0, 1.0, 0.25])
        # ponytail: bounded linear scan per bar; use a neighbor index if history grows substantially.
        for i in range(warmup - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            start = max(0, i - params.history_bars + 1)
            candidates = matured_features[start:i + 1]
            labels = outcomes[start:i + 1]
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(labels)
            if valid.sum() < 32:
                continue
            distances = ((candidates[valid] - features[i]) ** 2 * weights).sum(axis=1)
            neighbors = np.argsort(distances, kind="stable")[:9]
            forecast[i] = np.median(labels[valid][neighbors])
        out["forecast"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # This is a target position: stay long while the entry predicate
        # remains true and exit only when that same predicate becomes false.
        df["signal"] = (
            indicators["forecast"].notna()
            & (indicators["forecast"] > params.edge_threshold)
        ).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 600)))
    sample = pd.DataFrame(
        {"open": close * np.exp(rng.normal(0.0, 0.002, 600)),
         "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": np.full(600, 1000000.0)},
        index=pd.date_range("2020-01-01", periods=600, freq="B"),
    )
    strategy = GeneratedStrategy()
    params = ShapeParams()
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:400], params)
    pd.testing.assert_frame_equal(full.iloc[:400], prefix)
    signals = strategy.generate_signals(sample, full, None, params).data
    expected = (full["forecast"] > params.edge_threshold).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(signals["signal"], expected, check_names=False)
    assert signals["signal"].isin([0, 1]).all() and (signals["size"] > 0).all()
