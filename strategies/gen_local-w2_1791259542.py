from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    memory: int = 32
    smoothing: int = 5
    threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791259542"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        # Largest Haar support: 32 bars; correlation lag: 32 bars.
        return int(64 + params.memory + params.smoothing)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.memory < 2 or params.smoothing < 1 or not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("Require memory >= 2, smoothing >= 1, and a finite positive threshold")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_price"] = np.log(close.where(close.gt(0) & np.isfinite(close)))
        for scale in (4, 8, 16):
            mean = out["log_price"].rolling(scale, min_periods=scale).mean()
            wavelet = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            energy = wavelet.pow(2).rolling(params.memory, min_periods=params.memory).mean()
            # The lag separates the two complete Haar supports.
            correlation = wavelet.rolling(params.memory, min_periods=params.memory).corr(wavelet.shift(2 * scale))
            correlation = correlation.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
            normalized = wavelet / np.sqrt(energy.where(energy.gt(0)))
            out[f"mean_{scale}"] = mean
            out[f"wavelet_{scale}"] = wavelet
            out[f"energy_{scale}"] = energy
            out[f"correlation_{scale}"] = correlation
            out[f"normalized_{scale}"] = normalized
            out[f"contribution_{scale}"] = energy * correlation * normalized
        energies = [f"energy_{s}" for s in (4, 8, 16)]
        contributions = [f"contribution_{s}" for s in (4, 8, 16)]
        out["total_energy"] = out[energies].sum(axis=1, min_count=3)
        out["weighted_forecast"] = out[contributions].sum(axis=1, min_count=3) / out["total_energy"].where(out["total_energy"].gt(0))
        out["weighted_forecast"] = out["weighted_forecast"].replace([np.inf, -np.inf], np.nan)
        out["score"] = out["weighted_forecast"].rolling(params.smoothing, min_periods=params.smoothing).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan)
        trigger = np.select([score.ge(params.threshold), score.le(-params.threshold)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(trigger):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Exit flat; an opposite entry requires another qualifying close.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    scores = pd.DataFrame({"score": [np.nan, 0.4, 0.0, -0.4, -0.4, 0.0, 0.4, 0.4]}, index=index)
    params = WaveletParams()
    result = GeneratedStrategy.generate_signals(None, data, scores, None, params).data
    mirror = GeneratedStrategy.generate_signals(None, data, -scores, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert mirror["signal"].equals(-result["signal"])
    assert result["size"].gt(0).all()
