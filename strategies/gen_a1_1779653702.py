from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionReleaseParams:
    ma_period: int = 50
    regime_ma_period: int = 200
    zscore_window: int = 50
    compression_threshold: float = 0.5
    compression_bars: int = 10
    release_threshold: float = 1.0
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[CompressionReleaseParams]):
    strategy_id = "gen_a1_1779653702"

    @classmethod
    def params_type(cls):
        return CompressionReleaseParams

    @classmethod
    def warmup_bars(cls, params: CompressionReleaseParams) -> int:
        return (
            max(params.regime_ma_period, params.ma_period + params.zscore_window)
            + params.compression_bars
            + 5
        )

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: CompressionReleaseParams) -> pd.DataFrame:
        close = data["close"]

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        regime_ma = close.rolling(
            params.regime_ma_period, min_periods=params.regime_ma_period
        ).mean()

        # Normalised distance from the medium-term MA.
        distance = (close - ma) / ma.replace(0, np.nan)
        dist_std = distance.rolling(
            params.zscore_window, min_periods=params.zscore_window
        ).std()
        zscore = distance / dist_std.replace(0, np.nan)

        # Compression: |z| has been below the compression threshold every bar
        # for the last `compression_bars` bars.
        is_compressed = (zscore.abs() < params.compression_threshold).astype(float)
        compression_count = is_compressed.rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).sum()
        was_compressed = (compression_count >= params.compression_bars).astype(float)

        above_regime = (close > regime_ma).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["ma"] = ma
        ind["regime_ma"] = regime_ma
        ind["zscore"] = zscore
        ind["was_compressed"] = was_compressed
        ind["above_regime"] = above_regime
        return ind

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionReleaseParams,
    ) -> SignalFrame:
        z = indicators["zscore"]
        z_prev = z.shift(1)

        # Upward cross of the release threshold.
        release = (z > params.release_threshold) & (z_prev <= params.release_threshold)
        release = release.fillna(False)

        # Compression state must hold as of the bar BEFORE the release; the
        # release bar itself breaks compression by construction.
        was_compressed_prior = (
            indicators["was_compressed"].shift(1).fillna(0.0).astype(bool)
        )
        above_regime = indicators["above_regime"].fillna(0.0).astype(bool)

        entry = (release & was_compressed_prior & above_regime).to_numpy()

        # Fixed-bar exit: stay long for exactly `hold_bars` bars from entry.
        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        bars_remaining = 0
        hold = int(params.hold_bars) if params.hold_bars > 0 else 1
        for i in range(n):
            if bars_remaining > 0:
                raw_signal[i] = 1
                bars_remaining -= 1
            elif entry[i]:
                raw_signal[i] = 1
                bars_remaining = hold - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
