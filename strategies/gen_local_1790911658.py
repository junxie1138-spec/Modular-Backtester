from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    half_window: int = 4
    energy_window: int = 18
    polarization_threshold: float = 0.35
    coherence_threshold: float = 0.60

    def __post_init__(self):
        if self.half_window < 2 or self.energy_window < 2:
            raise ValueError("Wavelet windows must be at least two.")
        if not 0.0 < self.polarization_threshold < 1.0:
            raise ValueError("Polarization threshold must lie between zero and one.")
        if not 0.0 <= self.coherence_threshold < 1.0:
            raise ValueError("Coherence threshold must lie in [0, 1).")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790911658"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(8 * params.half_window + params.energy_window - 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for name, multiplier in (("fine", 1), ("medium", 2), ("coarse", 4)):
            half = params.half_window * multiplier
            out[name + "_mean"] = out["log_close"].rolling(half, min_periods=half).mean()
            # Causal Haar detail, normalized to comparable per-bar slopes.
            out[name] = (out[name + "_mean"] - out[name + "_mean"].shift(half)) / half
            out[name + "_energy"] = out[name].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[name + "_signed_energy"] = (out[name] * out[name].abs()).rolling(params.energy_window, min_periods=params.energy_window).mean()
        energies = ["fine_energy", "medium_energy", "coarse_energy"]
        signed = ["fine_signed_energy", "medium_signed_energy", "coarse_signed_energy"]
        out["total_energy"] = out[energies].sum(axis=1, min_count=3)
        out["signed_energy"] = out[signed].sum(axis=1, min_count=3)
        out["polarization"] = out["signed_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["cross_energy"] = (out["medium"] * out["coarse"]).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["phase_denominator"] = np.sqrt(out["medium_energy"] * out["coarse_energy"])
        out["coherence"] = (out["cross_energy"] / out["phase_denominator"].where(out["phase_denominator"] > 0)).clip(-1.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        coherent = indicators["coherence"] >= params.coherence_threshold
        long_entry = coherent & (indicators["polarization"] > params.polarization_threshold) & (indicators["medium"] > 0) & (indicators["coarse"] > 0)
        short_entry = coherent & (indicators["polarization"] < -params.polarization_threshold) & (indicators["medium"] < 0) & (indicators["coarse"] < 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, candidate in enumerate(candidates):
            if position != 0:
                if candidate == -position:
                    position = 0
            elif candidate != 0 and candidate == previous:
                position = int(candidate)
            signals[i] = position
            previous = int(candidate)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=100)
    close = pd.Series(100.0 * np.exp(np.arange(100) * 0.001), index=index)
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    features = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(features.iloc[:70], strategy.indicators(data.iloc[:70], params))
    assert strategy.warmup_bars(params) == 49
    assert features["polarization"].first_valid_index() == index[48]
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[:50].eq(0).all()
    assert result["signal"].iloc[50:].eq(1).all()
    assert result["size"].gt(0).all()
    polarity = np.array([1, 1, 0, -1, -1, -1, 0, 1, 1], dtype=float)
    fake = pd.DataFrame({"polarization": polarity, "coherence": 1.0, "medium": polarity, "coarse": polarity}, index=index[:9])
    signals = strategy.generate_signals(data.iloc[:9], fake, ctx, params).data["signal"].tolist()
    assert signals == [0, 0, 1, 1, 0, -1, -1, -1, 0]
    flat = data.copy()
    flat["close"] = 100.0
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
