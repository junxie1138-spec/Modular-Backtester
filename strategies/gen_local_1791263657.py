from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_bars: int = 5
    slow_bars: int = 40
    diffusion_bars: int = 5
    compression_ratio: float = 0.75
    diffusion_gain: float = 0.2
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791263657"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fast_bars, params.slow_bars, 2 * params.diffusion_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (1 <= params.fast_bars <= params.slow_bars and params.diffusion_bars >= 1
                and params.hold_bars >= 1 and 0 < params.compression_ratio <= 1
                and 0 <= params.diffusion_gain <= 1):
            raise ValueError("Invalid compression, diffusion, or holding parameters")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["return"] = data["close"].div(previous.where(previous > 0)).sub(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["fast_atr"] = out["true_range"].rolling(params.fast_bars, min_periods=params.fast_bars).mean()
        out["slow_atr"] = out["true_range"].rolling(params.slow_bars, min_periods=params.slow_bars).mean()
        out["compression"] = out["fast_atr"].div(out["slow_atr"].where(out["slow_atr"] > 0))
        out["up_session"] = out["return"].gt(0).astype(float).where(out["return"].notna())
        out["recent_share"] = out["up_session"].rolling(params.diffusion_bars, min_periods=params.diffusion_bars).mean()
        out["prior_share"] = out["recent_share"].shift(params.diffusion_bars)
        out["diffusion"] = out["recent_share"] - out["prior_share"]
        out["setup"] = (
            out["compression"].le(params.compression_ratio)
            & out["diffusion"].ge(params.diffusion_gain)
            & out["return"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        setup = indicators["setup"].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                signal[i] = int(remaining > 0)
            elif confirmed[i]:
                remaining = params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=16))
    indicators = pd.DataFrame({"setup": [False] + [True] * 15}, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0] + [1] * 8 + [0] + [1] * 4
    indicators["setup"] = [True, False] * 8
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].eq(0).all()
