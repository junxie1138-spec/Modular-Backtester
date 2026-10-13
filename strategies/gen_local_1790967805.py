from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_threshold: float = 0.65
    breakout_buffer: float = 0.001


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790967805"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not 0.0 < params.energy_threshold < 1.0:
            raise ValueError("energy_threshold must lie between zero and one")
        if not np.isfinite(params.breakout_buffer) or params.breakout_buffer < 0.0:
            raise ValueError("breakout_buffer must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        x = np.log(close.where(np.isfinite(close) & close.gt(0.0)))
        # Orthogonal Haar decomposition of the latest eight log closes.
        pair = (x - x.shift(1)) / np.sqrt(2.0)
        sum2 = x.rolling(2, min_periods=2).sum()
        quad = (sum2 - sum2.shift(2)) / 2.0
        sum4 = x.rolling(4, min_periods=4).sum()
        coarse = (sum4 - sum4.shift(4)) / np.sqrt(8.0)
        fine_energy = pair.pow(2)
        energy2 = fine_energy + fine_energy.shift(2) + fine_energy.shift(4) + fine_energy.shift(6)
        energy4 = quad.pow(2) + quad.shift(4).pow(2)
        energy8 = coarse.pow(2)
        total = energy2 + energy4 + energy8
        out["energy2"] = energy2.fillna(0.0)
        out["energy4"] = energy4.fillna(0.0)
        out["energy8"] = energy8.fillna(0.0)
        out["signed_energy"] = (np.sign(coarse) * energy8 / total.where(total.gt(0.0))).fillna(0.0)
        upper = data["high"].where(np.isfinite(data["high"])).rolling(8, min_periods=8).max().shift(1)
        lower = data["low"].where(np.isfinite(data["low"])).rolling(8, min_periods=8).min().shift(1)
        ready = total.notna() & upper.notna() & lower.notna() & np.isfinite(close) & close.gt(0.0)
        out["upper"] = upper.fillna(0.0)
        out["lower"] = lower.fillna(0.0)
        out["ready"] = ready
        out["bullish"] = (ready & out["signed_energy"].ge(params.energy_threshold) & close.gt(upper * (1.0 + params.breakout_buffer))).fillna(False)
        out["bearish"] = (ready & out["signed_energy"].le(-params.energy_threshold) & close.lt(lower / (1.0 + params.breakout_buffer))).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        # Hysteresis: neutral observations retain the previous breakout regime.
        # Only the symmetric bearish condition flips an established long to flat.
        regime = pd.Series(np.nan, index=data.index, dtype=float)
        regime.loc[indicators["bullish"].fillna(False)] = 1.0
        regime.loc[indicators["bearish"].fillna(False)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = regime.ffill().fillna(0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    close = pd.Series(np.exp(np.arange(20) * 0.01) * 100.0, index=index)
    data = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000.0}, index=index)
    features = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, features, None, params).data
    assert strategy.warmup_bars(params) == 9
    assert not features.isna().any().any()
    assert signals["signal"].iloc[:9].eq(0).all()
    assert signals["signal"].iloc[9:].eq(1).all()
    events = pd.DataFrame({"bullish": [False, True, False, False, False, False], "bearish": [False, False, False, False, True, False]}, index=index[:6])
    result = strategy.generate_signals(data.iloc[:6], events, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0.0).all()
