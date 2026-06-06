from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicGrowthParams:
    infection_window: int = 20
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[EpidemicGrowthParams]):
    strategy_id = "gen_a1_1778883103"

    _ATR_PERIOD = 14
    _MAX_HOLD = 5
    _SATURATION = 0.5

    @classmethod
    def params_type(cls) -> type[EpidemicGrowthParams]:
        return EpidemicGrowthParams

    @staticmethod
    def warmup_bars(params: EpidemicGrowthParams) -> int:
        w = max(int(params.infection_window), 2)
        return max(w, GeneratedStrategy._ATR_PERIOD) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: EpidemicGrowthParams) -> pd.DataFrame:
        w = max(int(params.infection_window), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ret = close.pct_change()
        vol_avg = volume.rolling(w).mean()
        # "new infection": a volume-confirmed up move on this bar.
        infected = ((ret > 0.0) & (volume > vol_avg)).astype(float)
        # prevalence = infected fraction of the population (rolling window).
        prevalence = infected.rolling(w).mean()
        prevalence_prev = prevalence.shift(1)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(GeneratedStrategy._ATR_PERIOD).mean()

        # SI mass-action: transmission force p*(1-p) is largest mid-curve. The
        # growth phase is prevalence rising while still below the saturation
        # midpoint (large susceptible pool). Saturated => pool exhausted =>
        # the move is mature; no fresh entries.
        growth_regime = (prevalence < GeneratedStrategy._SATURATION) & (
            prevalence > prevalence_prev
        )
        entry_trigger = (infected > 0.5) & growth_regime

        out = pd.DataFrame(index=data.index)
        out["close"] = close
        out["atr"] = atr
        out["prevalence"] = prevalence.fillna(0.0)
        out["entry_trigger"] = entry_trigger.fillna(False).astype(float)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = indicators["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        trigger = indicators["entry_trigger"].to_numpy(dtype=float) > 0.5
        n = len(close)
        raw = np.zeros(n, dtype=int)

        k = float(params.atr_mult)
        max_hold = GeneratedStrategy._MAX_HOLD

        in_pos = False
        high_water = 0.0
        bars_held = 0

        for i in range(n):
            atr_i = atr[i]
            atr_ok = bool(np.isfinite(atr_i)) and atr_i > 0.0
            if not in_pos:
                if trigger[i] and atr_ok:
                    in_pos = True
                    high_water = close[i]
                    bars_held = 0
                    raw[i] = 1
            else:
                bars_held += 1
                if close[i] > high_water:
                    high_water = close[i]
                # Rolling-high trailing stop: ratchets up with high_water,
                # never down. Exit when close falls k*ATR below the in-trade
                # high-water mark.
                stop_level = high_water - k * atr_i if atr_ok else high_water
                exit_now = (
                    (not atr_ok)
                    or (close[i] <= stop_level)
                    or (bars_held >= max_hold)
                )
                if exit_now:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
