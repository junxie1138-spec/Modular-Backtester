from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalVolParams:
    atr_period: int = 14
    vol_med_period: int = 63
    tidal_annual_bars: int = 252
    tidal_smooth_window: int = 5
    k_stop: float = 2.0
    max_hold: int = 5
    atr_ratio_threshold: float = 0.95


class GeneratedStrategy(BaseStrategy[TidalVolParams]):
    strategy_id = "gen_jay_1778898128"

    @classmethod
    def params_type(cls) -> type[TidalVolParams]:
        return TidalVolParams

    @staticmethod
    def warmup_bars(params: TidalVolParams) -> int:
        return params.atr_period + params.vol_med_period + params.tidal_smooth_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: TidalVolParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        atr_median = atr.rolling(params.vol_med_period, min_periods=params.vol_med_period).median()
        atr_ratio = atr / atr_median

        doy = pd.Series(data.index.dayofyear, index=data.index, dtype=float)
        raw_phase = np.sin(2.0 * np.pi * doy / float(params.tidal_annual_bars))
        tidal_phase = raw_phase.rolling(params.tidal_smooth_window, min_periods=1).mean()

        phase_delta = tidal_phase - tidal_phase.shift(1)
        tidal_ascending = (phase_delta > 0).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["atr_ratio"] = atr_ratio
        ind["tidal_phase"] = tidal_phase
        ind["tidal_ascending"] = tidal_ascending
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalVolParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        atr_ratio = indicators["atr_ratio"].to_numpy()
        tidal_ascending = indicators["tidal_ascending"].to_numpy()

        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        cond_compressed = atr_ratio < params.atr_ratio_threshold
        cond_ascending = tidal_ascending == 1.0
        raw_cond = (cond_compressed & cond_ascending).astype(np.int8)

        prev_cond = np.zeros(n, dtype=np.int8)
        prev_cond[1:] = raw_cond[:-1]
        two_bar = (raw_cond == 1) & (prev_cond == 1)

        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if in_position:
                bars_held += 1
                stop_level = entry_price - params.k_stop * entry_atr
                if close[i] < stop_level or bars_held >= params.max_hold:
                    in_position = False
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                else:
                    signal[i] = 1
            else:
                if two_bar[i] and not np.isnan(atr[i]) and not np.isnan(atr_ratio[i]):
                    signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
