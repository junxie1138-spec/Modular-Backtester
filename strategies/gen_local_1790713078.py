from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 32
    hold_bars: int = 15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713078"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.lookback + 1, 2 ** 3 + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ret = np.log(data["close"] / data["close"].shift(1))
        # Haar detail energy at dyadic scales 2, 4, and 8 bars.
        energies = []
        for scale in (2, 4, 8):
            half = scale // 2
            detail = ret.rolling(half).mean() - ret.shift(half).rolling(half).mean()
            energies.append(detail.pow(2))
        multiscale_energy = energies[1] + energies[2]
        imbalance = multiscale_energy / (energies[0] + multiscale_energy + 1e-12)
        low = data["low"].rolling(params.lookback).min()
        high = data["high"].rolling(params.lookback).max()
        position = (data["close"] - low) / (high - low).replace(0, np.nan)
        return pd.DataFrame({"energy_bias": imbalance, "range_position": position}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        ready = indicators["energy_bias"].notna() & indicators["range_position"].notna()
        entries = (ready & (indicators["energy_bias"] > 0.72) & (indicators["range_position"] < 0.25)).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        active_until = -1
        for i, enter in enumerate(entries):
            if i > active_until and enter:
                raw[i] = 1
                active_until = i + params.hold_bars - 1
        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
