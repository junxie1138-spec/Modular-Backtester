from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPositionParams:
    position_lookback: int = 8
    energy_window: int = 4
    energy_threshold: float = 0.50
    edge: float = 0.20
    profit_target: float = 0.007
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[WaveletPositionParams]):
    strategy_id = "gen_local-w2_1791222962"

    @classmethod
    def params_type(cls) -> type[WaveletPositionParams]:
        return WaveletPositionParams

    @classmethod
    def warmup_bars(cls, params: WaveletPositionParams) -> int:
        return max(params.position_lookback + 1, params.energy_window + 4)

    @classmethod
    def indicators(
        cls, data: pd.DataFrame, params: WaveletPositionParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change().fillna(0.0)

        d1 = (returns - returns.shift(1)) / np.sqrt(2.0)
        d2 = (
            returns
            + returns.shift(1)
            - returns.shift(2)
            - returns.shift(3)
        ) / 2.0
        approximation = (
            returns
            + returns.shift(1)
            + returns.shift(2)
            + returns.shift(3)
        ) / 2.0

        fine_energy = d1.pow(2).rolling(params.energy_window).mean()
        middle_energy = d2.pow(2).rolling(params.energy_window).mean()
        trend_energy = approximation.pow(2).rolling(params.energy_window).mean()
        total_energy = fine_energy + middle_energy + trend_energy
        energy_ratio = trend_energy.div(total_energy.replace(0.0, np.nan))

        rolling_low = data["low"].rolling(params.position_lookback).min()
        rolling_high = data["high"].rolling(params.position_lookback).max()
        span = (rolling_high - rolling_low).replace(0.0, np.nan)
        relative_position = (close - rolling_low).div(span)

        return pd.DataFrame(
            {
                "relative_position": relative_position.fillna(0.5).clip(0.0, 1.0),
                "energy_ratio": energy_ratio.fillna(0.0).clip(0.0, 1.0),
                "wavelet_direction": approximation.fillna(0.0),
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletPositionParams,
    ) -> SignalFrame:
        position = indicators["relative_position"]
        energy = indicators["energy_ratio"]
        direction = indicators["wavelet_direction"]

        candidate = pd.Series(0, index=data.index, dtype=int)
        active = energy >= params.energy_threshold
        candidate.loc[
            active & (position >= 1.0 - params.edge) & (direction > 0.0)
        ] = 1
        candidate.loc[
            active & (position <= params.edge) & (direction < 0.0)
        ] = -1

        close = data["close"].astype(float).to_numpy()
        entries = candidate.to_numpy()
        signals = np.zeros(len(data), dtype=int)
        current = 0
        entry_price = 0.0
        held = 0

        for i in range(len(data)):
            if current == 0:
                if entries[i] != 0:
                    current = int(entries[i])
                    entry_price = close[i]
                    held = 0
            else:
                held += 1
                gain = current * (close[i] / entry_price - 1.0)
                if gain >= params.profit_target or held >= params.max_hold:
                    current = 0
                    held = 0
            signals[i] = current

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = (0.5 + 0.5 * energy).clip(0.5, 1.0).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
