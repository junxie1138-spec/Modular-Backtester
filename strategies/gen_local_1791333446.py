from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    latent_window: int = 5
    baseline_window: int = 8
    wick_window: int = 5
    range_window: int = 20
    compression_threshold: float = 0.9
    innovation_threshold: float = 0.03
    wick_threshold: float = 0.06


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791333446"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(params.range_window, params.latent_window + params.baseline_window + 1, 2 * params.wick_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.latent_window, params.baseline_window, params.wick_window, params.range_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        thresholds = np.array([params.compression_threshold, params.innovation_threshold, params.wick_threshold])
        if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
            raise ValueError("Thresholds must be finite and positive")
        d = pd.DataFrame(index=data.index)
        d["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        d["fast_range"] = d["range"].rolling(params.latent_window).mean()
        d["slow_range"] = d["range"].rolling(params.range_window).mean()
        d["compression"] = d["fast_range"] / d["slow_range"]
        d["latent_close"] = data["close"].rolling(params.latent_window).mean()
        d["latent_slope"] = d["latent_close"].diff()
        d["expected_slope"] = d["latent_slope"].shift(1).rolling(params.baseline_window).mean()
        d["innovation"] = (d["latent_slope"] - d["expected_slope"]) / d["slow_range"]
        body_top = data[["open", "close"]].max(axis=1, skipna=False)
        body_bottom = data[["open", "close"]].min(axis=1, skipna=False)
        d["upper_wick"] = (data["high"] - body_top).clip(lower=0) / d["range"]
        d["lower_wick"] = (body_bottom - data["low"]).clip(lower=0) / d["range"]
        d["wick_balance"] = d["lower_wick"] - d["upper_wick"]
        d["recent_balance"] = d["wick_balance"].rolling(params.wick_window).mean()
        d["prior_balance"] = d["recent_balance"].shift(params.wick_window)
        d["wick_release"] = d["recent_balance"] - d["prior_balance"]
        return d.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        features = indicators[["compression", "innovation", "wick_release"]]
        valid = pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=data.index)
        compressed = valid & indicators["compression"].le(params.compression_threshold)
        buy = compressed & indicators["innovation"].gt(params.innovation_threshold) & indicators["wick_release"].gt(params.wick_threshold)
        sell = compressed & indicators["innovation"].lt(-params.innovation_threshold) & indicators["wick_release"].lt(-params.wick_threshold)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (long_entry, short_entry) in enumerate(zip(buy.to_numpy(), sell.to_numpy())):
            if (position == 1 and short_entry) or (position == -1 and long_entry):
                position = 0
            elif position == 0:
                position = 1 if long_entry else -1 if short_entry else 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    direction = np.array([1, 0, -1, -1, 0, 1, 1], dtype=float)
    indicators = pd.DataFrame({"compression": 0.8, "innovation": direction, "wick_release": direction}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, CompressionParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    indicators.loc[:, "innovation"] = np.nan
    result = GeneratedStrategy.generate_signals(data, indicators, None, CompressionParams()).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
