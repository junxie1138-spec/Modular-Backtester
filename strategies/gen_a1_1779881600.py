from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StandingWaveTideParams:
    wave_window: int = 15
    pctile_lookback: int = 252
    atr_period: int = 14
    trend_ma: int = 200
    amplitude_pctile_max: float = 0.20
    position_in_range_max: float = 0.50
    trail_atr_mult: float = 2.5
    min_size: float = 0.40
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[StandingWaveTideParams]):
    strategy_id = "gen_a1_1779881600"

    @classmethod
    def params_type(cls):
        return StandingWaveTideParams

    @classmethod
    def warmup_bars(cls, params: StandingWaveTideParams) -> int:
        return int(
            max(
                params.pctile_lookback,
                params.trend_ma,
                params.wave_window,
                params.atr_period,
            )
            + 2
        )

    def indicators(self, data: pd.DataFrame, params: StandingWaveTideParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        rolling_hi = high.rolling(params.wave_window, min_periods=params.wave_window).max()
        rolling_lo = low.rolling(params.wave_window, min_periods=params.wave_window).min()
        wave_amp = rolling_hi - rolling_lo

        amp_pctile = wave_amp.rolling(
            params.pctile_lookback, min_periods=params.pctile_lookback
        ).rank(pct=True)

        range_denom = (rolling_hi - rolling_lo).replace(0.0, np.nan)
        pos_in_range = (close - rolling_lo) / range_denom
        pos_in_range = pos_in_range.clip(lower=0.0, upper=1.0)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        trend_ma = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()
        above_trend = (close > trend_ma).astype(int)

        compression = (1.0 - amp_pctile).clip(lower=0.0, upper=1.0)
        floor_prox = (1.0 - pos_in_range).clip(lower=0.0, upper=1.0)
        strength = compression * floor_prox

        out = pd.DataFrame(
            {
                "wave_amp": wave_amp,
                "amp_pctile": amp_pctile,
                "pos_in_range": pos_in_range,
                "atr": atr,
                "trend_ma": trend_ma,
                "above_trend": above_trend,
                "strength": strength,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StandingWaveTideParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=np.float64)
        atr = indicators["atr"].to_numpy(dtype=np.float64)
        amp_pctile = indicators["amp_pctile"].to_numpy(dtype=np.float64)
        pos_in_range = indicators["pos_in_range"].to_numpy(dtype=np.float64)
        above_trend = indicators["above_trend"].to_numpy(dtype=np.int64)
        strength = indicators["strength"].to_numpy(dtype=np.float64)

        n = len(close)
        sig = np.zeros(n, dtype=np.int64)
        sz = np.zeros(n, dtype=np.float64)

        in_pos = False
        hwm = 0.0
        active_size = 0.0

        amp_thresh = float(params.amplitude_pctile_max)
        pos_thresh = float(params.position_in_range_max)
        k = float(params.trail_atr_mult)
        min_sz = float(params.min_size)
        max_sz = float(params.max_size)
        if max_sz < min_sz:
            max_sz = min_sz
        size_span = max_sz - min_sz

        for i in range(n):
            c = close[i]
            a = atr[i]

            if in_pos:
                if c > hwm:
                    hwm = c
                if (not np.isnan(a)) and c <= hwm - k * a:
                    in_pos = False
                    sig[i] = 0
                    sz[i] = 0.0
                    continue
                sig[i] = 1
                sz[i] = active_size
                continue

            if (
                (not np.isnan(amp_pctile[i]))
                and (not np.isnan(pos_in_range[i]))
                and (not np.isnan(a))
                and amp_pctile[i] <= amp_thresh
                and pos_in_range[i] <= pos_thresh
                and above_trend[i] == 1
            ):
                s = strength[i]
                if np.isnan(s) or s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
                scaled = min_sz + size_span * s
                if scaled < min_sz:
                    scaled = min_sz
                elif scaled > max_sz:
                    scaled = max_sz
                in_pos = True
                hwm = c
                active_size = scaled
                sig[i] = 1
                sz[i] = scaled

        df = pd.DataFrame({"signal": sig, "size": sz}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
