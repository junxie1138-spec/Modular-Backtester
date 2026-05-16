from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    obv_trend_span: int = 20
    obv_noise_window: int = 20
    snr_lookback: int = 60
    snr_pct_threshold: float = 0.55
    vol_window: int = 15
    vol_multiplier: float = 1.25
    close_pct_threshold: float = 0.6
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778905664"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.snr_lookback + params.obv_noise_window + params.obv_trend_span + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]

        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()

        obv_trend = obv.ewm(span=params.obv_trend_span, adjust=False).mean()
        obv_residual = obv - obv_trend
        obv_noise = obv_residual.rolling(params.obv_noise_window).std()
        obv_signal_strength = obv_trend.diff(params.obv_trend_span).abs()
        snr = obv_signal_strength / (obv_noise + 1e-9)

        snr_rank = snr.rolling(params.snr_lookback).rank(pct=True)

        vol_mean = volume.rolling(params.vol_window).mean()
        vol_ratio = volume / (vol_mean + 1e-9)

        bar_range = data["high"] - data["low"]
        close_position = (close - data["low"]) / (bar_range + 1e-9)

        ind["snr_rank"] = snr_rank
        ind["vol_ratio"] = vol_ratio
        ind["close_position"] = close_position

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        # Primitive 1: OBV SNR regime — high signal-to-noise = clean directional accumulation
        regime_on = indicators["snr_rank"] >= params.snr_pct_threshold

        # Primitive 2: volume-confirmed bullish move — spike volume AND close in upper portion of bar
        vol_burst = indicators["vol_ratio"] >= params.vol_multiplier
        bullish_close = indicators["close_position"] >= params.close_pct_threshold
        vol_confirmed = vol_burst & bullish_close

        # Two-primitive AND gate
        raw_entry = regime_on & vol_confirmed

        # Fixed-bar exit loop — path-dependent, no vectorised equivalent
        n = params.hold_bars
        signal_arr = np.zeros(len(df), dtype=int)
        in_trade = False
        bars_held = 0

        for i in range(len(df)):
            if in_trade:
                bars_held += 1
                if bars_held >= n:
                    in_trade = False
                    bars_held = 0
                else:
                    signal_arr[i] = 1
            else:
                if bool(raw_entry.iloc[i]):
                    in_trade = True
                    bars_held = 0
                    signal_arr[i] = 1

        df["signal"] = signal_arr
        # Shift: decision on bar N closes, fill on bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
