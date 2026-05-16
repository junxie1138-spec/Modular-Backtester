from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolStabilityCoilParams:
    atr_window: int = 14
    vov_window: int = 20
    trend_window: int = 100
    entry_thr: float = 0.12
    exit_thr: float = 0.22
    spike_mult: float = 2.2
    refractory_bars: int = 5
    base_size: float = 0.4
    size_scale: float = 0.6
    min_size: float = 0.2
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[VolStabilityCoilParams]):
    strategy_id = "gen_a1_1778910288"

    @classmethod
    def params_type(cls) -> type[VolStabilityCoilParams]:
        return VolStabilityCoilParams

    def warmup_bars(self, params: VolStabilityCoilParams) -> int:
        # ATR needs atr_window bars; vov is a rolling stat over ATR -> add vov_window.
        # trend SMA needs trend_window bars. Take the longest plus a safety buffer.
        return int(max(params.atr_window + params.vov_window, params.trend_window) + 5)

    def indicators(self, data: pd.DataFrame, params: VolStabilityCoilParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True range (uses prior close -> first bar is NaN, handled below).
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Vol-of-vol: coefficient of variation of ATR over vov_window.
        atr_mean = atr.rolling(params.vov_window, min_periods=params.vov_window).mean()
        atr_std = atr.rolling(params.vov_window, min_periods=params.vov_window).std()
        vov = atr_std / atr_mean.replace(0.0, np.nan)

        trend_sma = close.rolling(params.trend_window, min_periods=params.trend_window).mean()
        uptrend = (close > trend_sma).fillna(False)

        # Volatility spike: a true range that dwarfs the prior ATR.
        spike = (tr > params.spike_mult * atr.shift(1)).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["atr"] = atr
        out["vov"] = vov
        out["trend_sma"] = trend_sma
        out["uptrend"] = uptrend.astype(float)
        out["spike"] = spike.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolStabilityCoilParams,
    ) -> SignalFrame:
        n = len(data)
        vov = indicators["vov"].to_numpy(dtype=float)
        uptrend = indicators["uptrend"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=int)
        size = np.full(n, params.min_size, dtype=float)

        entry_thr = float(params.entry_thr)
        exit_thr = float(params.exit_thr)
        ref_max = int(params.refractory_bars)
        min_size = float(params.min_size)
        max_size = float(params.max_size)

        in_pos = False
        refractory = 0
        held_size = min_size

        for i in range(n):
            v = vov[i]
            valid = not np.isnan(v)
            up = uptrend[i] > 0.5
            is_spike = spike[i] > 0.5

            if in_pos:
                # Signal-reversal exit: the entry condition was "vol is stable AND
                # uptrend". Exit only when that flips - vol destabilizes (vov above
                # the hysteresis exit band) or the uptrend gate breaks.
                if (not valid) or (v > exit_thr) or (not up):
                    in_pos = False
                    sig[i] = 0
                    size[i] = min_size
                else:
                    sig[i] = 1
                    size[i] = held_size
            else:
                # Entry: equilibrium coil (low vov), uptrend, and no recent spike.
                if valid and refractory == 0 and v < entry_thr and up:
                    strength = (entry_thr - v) / entry_thr
                    if strength < 0.0:
                        strength = 0.0
                    elif strength > 1.0:
                        strength = 1.0
                    # Signal-scaled sizing: flatter vol -> stronger signal -> bigger size.
                    s = params.base_size + params.size_scale * strength
                    if s < min_size:
                        s = min_size
                    elif s > max_size:
                        s = max_size
                    in_pos = True
                    held_size = s
                    sig[i] = 1
                    size[i] = s
                else:
                    sig[i] = 0
                    size[i] = min_size

            # Refractory lockout: a volatility spike blocks NEW entries for
            # ref_max bars. It never force-exits an open position - the vov
            # exit band handles that, keeping the exit purely signal-reversal.
            if is_spike:
                refractory = ref_max
            elif refractory > 0:
                refractory -= 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        # Mandatory one-bar shift: decide on bar N's close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
