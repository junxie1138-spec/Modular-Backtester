from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPersistenceParams:
    percentile_window: int = 126
    energy_quantile: float = 0.75
    stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletPersistenceParams]):
    strategy_id = "gen_local_1791224194"

    @classmethod
    def params_type(cls) -> type[WaveletPersistenceParams]:
        return WaveletPersistenceParams

    @classmethod
    def warmup_bars(cls, params: WaveletPersistenceParams) -> int:
        return params.percentile_window + 32

    def indicators(self, data: pd.DataFrame, params: WaveletPersistenceParams) -> pd.DataFrame:
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        details = {}
        for scale in (2, 4, 8):
            details[scale] = returns.rolling(scale).mean() - returns.shift(scale).rolling(scale).mean()

        energy = {scale: details[scale].pow(2) for scale in details}
        energy_share = (energy[4] + energy[8]) / (energy[2] + energy[4] + energy[8] + 1e-12)
        lagged = details[4].shift(4)
        numerator = (details[4] * lagged).rolling(16).sum()
        denominator = np.sqrt(details[4].pow(2).rolling(16).sum() * lagged.pow(2).rolling(16).sum())
        persistence = numerator / (denominator + 1e-12)

        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        result = pd.DataFrame(index=data.index)
        result["detail_4"] = details[4]
        result["detail_8"] = details[8]
        result["energy_share"] = energy_share
        result["energy_cutoff"] = energy_share.rolling(params.percentile_window).quantile(params.energy_quantile)
        result["persistence"] = persistence
        result["persistence_cutoff"] = persistence.rolling(params.percentile_window).quantile(0.60)
        result["atr"] = true_range.rolling(14).mean()
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletPersistenceParams,
    ) -> SignalFrame:
        ready = np.arange(len(data)) >= self.warmup_bars(params)
        candidate = (
            ready
            & (indicators["detail_4"].to_numpy() > 0)
            & (indicators["detail_8"].to_numpy() > 0)
            & (indicators["energy_share"].to_numpy() > indicators["energy_cutoff"].to_numpy())
            & (indicators["persistence"].to_numpy() > indicators["persistence_cutoff"].to_numpy())
            & (indicators["atr"].to_numpy() > 0)
        )
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = None
        stop_distance = 0.0
        held_bars = 0

        for i in range(len(data)):
            if active:
                if entry_price is None:
                    entry_price = float(opens[i])
                held_bars += 1
                if closes[i] < entry_price - stop_distance or held_bars >= params.max_hold:
                    active = False
                    entry_price = None
                else:
                    signal[i] = 1
            elif candidate[i]:
                active = True
                stop_distance = params.stop_atr * float(atr[i])
                held_bars = 0
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
