from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicTrendParams:
    prevalence_window: int = 5
    vol_ma_window: int = 20
    vol_mult: float = 1.0
    beta: float = 4.0
    f_enter: float = 0.55
    f_exit: float = 0.30
    rise_lag: int = 2
    sma_window: int = 50
    size_min: float = 0.5
    size_max: float = 1.5


class GeneratedStrategy(BaseStrategy[EpidemicTrendParams]):
    """Trend-strength entry driven by an SI-epidemic force-of-infection term.

    Each bar that closes up on above-average volume is an 'infected' bar. The
    infected fraction over a short window is the prevalence I; S = 1 - I is the
    susceptible fraction. The force of infection F = beta * S * I peaks mid
    epidemic and is high only when a trend is actively recruiting participants.
    Entry fires on the epidemic upslope (F high AND prevalence rising); the exit
    is a pure signal-reversal flip of that same F regime via hysteresis.
    """

    strategy_id = "gen_a1_1778884346"

    @classmethod
    def params_type(cls) -> type[EpidemicTrendParams]:
        return EpidemicTrendParams

    @staticmethod
    def warmup_bars(params: EpidemicTrendParams) -> int:
        base = max(
            int(params.vol_ma_window) + int(params.prevalence_window),
            int(params.sma_window),
        )
        return int(base + int(params.rise_lag) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EpidemicTrendParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        prev_window = max(int(params.prevalence_window), 1)
        vol_window = max(int(params.vol_ma_window), 1)
        sma_window = max(int(params.sma_window), 1)

        up_close = close > close.shift(1)
        vol_ma = volume.rolling(vol_window, min_periods=vol_window).mean()
        vol_ok = volume > (vol_ma * float(params.vol_mult))
        confirmed_up = (up_close & vol_ok).astype(float)

        # Epidemic prevalence: infected fraction of recent bars.
        infected = confirmed_up.rolling(
            prev_window, min_periods=prev_window
        ).mean()
        susceptible = 1.0 - infected
        # Force of infection F = beta * S * I, maximal at prevalence 0.5.
        force = float(params.beta) * susceptible * infected

        sma = close.rolling(sma_window, min_periods=sma_window).mean()
        trend_ok = (close >= sma).astype(float)

        out = pd.DataFrame(index=data.index)
        out["infected"] = infected
        out["force"] = force
        out["trend_ok"] = trend_ok
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EpidemicTrendParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        infected = indicators["infected"].to_numpy(dtype=float)
        force = indicators["force"].to_numpy(dtype=float)
        trend_ok = indicators["trend_ok"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        lag = max(int(params.rise_lag), 1)
        f_enter = float(params.f_enter)
        f_exit = float(params.f_exit)
        smin = float(params.size_min)
        smax = float(params.size_max)
        span = max(1.0 - f_enter, 1e-6)

        in_pos = False
        for i in range(n):
            f_i = force[i]
            inf_i = infected[i]
            valid = (
                np.isfinite(f_i)
                and np.isfinite(inf_i)
                and trend_ok[i] == 1.0
            )

            if i - lag >= 0 and np.isfinite(infected[i - lag]):
                rising = inf_i > infected[i - lag]
            else:
                rising = False

            if not valid:
                in_pos = False
                signal[i] = 0
                continue

            if in_pos:
                # Signal-reversal exit: the force-of-infection regime that
                # justified the entry has flipped off (hysteresis band).
                if f_i < f_exit:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                # Enter on the epidemic upslope: high force AND rising prevalence.
                if f_i >= f_enter and rising:
                    in_pos = True
                    signal[i] = 1
                else:
                    signal[i] = 0

            if signal[i] == 1:
                # Signal-scaled sizing: stronger force -> larger position.
                strength = (min(f_i, 1.0) - f_enter) / span
                strength = min(max(strength, 0.0), 1.0)
                size[i] = smin + (smax - smin) * strength

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=0.01)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
