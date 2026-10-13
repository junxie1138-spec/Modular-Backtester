from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    smooth_bars: int = 4
    baseline_bars: int = 12
    noise_bars: int = 20
    channel_bars: int = 30
    hold_bars: int = 8
    innovation_threshold: float = 0.9
    min_efficiency: float = 0.15

    def __post_init__(self):
        if min(self.smooth_bars, self.baseline_bars, self.noise_bars, self.channel_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if self.hold_bars < 1 or not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0:
            raise ValueError("Holding period and innovation threshold must be positive")
        if not 0 <= self.min_efficiency <= 1:
            raise ValueError("Efficiency must lie between zero and one")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791321812"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.channel_bars, 2 * params.smooth_bars + params.baseline_bars + params.noise_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["delta"] = out["log_close"].diff()
        out["latent_level"] = out["log_close"].rolling(params.smooth_bars).mean()
        out["slope"] = out["latent_level"].diff().rolling(params.smooth_bars).mean()
        out["expected_slope"] = out["slope"].shift(1).rolling(params.baseline_bars).mean()
        out["innovation"] = out["slope"] - out["expected_slope"]
        out["innovation_noise"] = out["innovation"].rolling(params.noise_bars).std(ddof=0).shift(1)
        out["z"] = out["innovation"] / out["innovation_noise"].where(out["innovation_noise"] > 1e-12)
        out["lower"] = data["low"].rolling(params.channel_bars).min()
        out["upper"] = data["high"].rolling(params.channel_bars).max()
        out["width"] = out["upper"] - out["lower"]
        out["position"] = ((data["close"] - out["lower"]) / out["width"].where(out["width"] > 0)).clip(0, 1)
        out["net_move"] = out["delta"].rolling(params.noise_bars).sum().abs()
        out["path_length"] = out["delta"].abs().rolling(params.noise_bars).sum()
        out["efficiency"] = (out["net_move"] / out["path_length"].where(out["path_length"] > 1e-12)).clip(0, 1)
        out["strength"] = out["z"].abs() * (1 + 2 * (out["position"] - 0.5).abs())
        out["ready"] = np.isfinite(out).all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        eligible = indicators["ready"].astype(bool) & (indicators["efficiency"] >= params.min_efficiency)
        long_entry = eligible & (indicators["z"] >= params.innovation_threshold) & (indicators["slope"] > 0) & (indicators["position"] < 0.5)
        short_entry = eligible & (indicators["z"] <= -params.innovation_threshold) & (indicators["slope"] < 0) & (indicators["position"] > 0.5)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        entry_sizes = (0.25 + 0.75 * (indicators["strength"] / (3 * params.innovation_threshold)).clip(0, 1)).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position, entry_bar, held_size = 0, -1, 1.0
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(warmup, len(data)):
            if position:
                if i - entry_bar >= params.hold_bars:
                    position, held_size = 0, 1.0
                    # Exit bar stays flat; entry signals cannot cancel the timed exit.
                signals[i], sizes[i] = position, held_size
            elif candidates[i]:
                position, entry_bar, held_size = int(candidates[i]), i, float(entry_sizes[i])
                signals[i], sizes[i] = position, held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    w, h = GeneratedStrategy.warmup_bars(params), params.hold_bars
    index = pd.date_range("2020-01-01", periods=w + 2 * h + 5, freq="B")
    prices = pd.Series(100 * np.exp(np.arange(len(index)) * 0.001), index=index)
    data = pd.DataFrame({"open": prices, "high": prices * 1.01, "low": prices * 0.99, "close": prices, "volume": 1000000.0})
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(index) and np.isfinite(derived).all().all()
    z = np.where(np.arange(len(index)) <= w, 1.5, -2.5)
    fixture = pd.DataFrame({"ready": True, "efficiency": 1.0, "z": z, "slope": np.sign(z), "position": np.where(z > 0, 0.25, 0.75), "strength": np.abs(z) * 1.5}, index=index)
    result = GeneratedStrategy.generate_signals(data, fixture, None, params).data
    assert (result["signal"].iloc[:w + 1] == 0).all()
    assert (result["signal"].iloc[w + 1:w + h + 1] == 1).all()
    assert result["signal"].iloc[w + h + 1] == 0
    assert (result["signal"].iloc[w + h + 2:w + 2 * h + 2] == -1).all()
    assert result["signal"].iloc[w + 2 * h + 2] == 0
    assert result["size"].iloc[w + 1:w + h + 1].nunique() == 1
    assert result["size"].iloc[w + h + 2] > result["size"].iloc[w + 1]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
