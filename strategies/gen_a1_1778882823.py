from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicStreakParams:
    prevalence_window: int = 20
    susceptible_thresh: float = 0.50
    entry_streak: int = 3
    outbreak_extra: int = 2
    vol_ma_window: int = 20
    vol_burst_mult: float = 1.20
    profit_target: float = 0.015
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[EpidemicStreakParams]):
    strategy_id = "gen_a1_1778882823"

    @classmethod
    def params_type(cls):
        return EpidemicStreakParams

    def warmup_bars(self, params: EpidemicStreakParams) -> int:
        return int(max(int(params.prevalence_window),
                       int(params.vol_ma_window),
                       int(params.entry_streak) + int(params.outbreak_extra))) + 2

    def indicators(self, data: pd.DataFrame, params: EpidemicStreakParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        pw = max(int(params.prevalence_window), 2)
        vw = max(int(params.vol_ma_window), 2)

        # Per-bar 'infection': a lower close than the prior bar.
        is_down = (close < close.shift(1))
        is_down = is_down.fillna(False)

        # Epidemic prevalence: fraction of recent bars currently infected.
        prevalence = is_down.astype(float).rolling(window=pw, min_periods=pw).mean()

        # Consecutive-streak count: length of the active down-close run.
        down_int = is_down.astype(int)
        reset_grp = (~is_down).cumsum()
        down_streak = down_int.groupby(reset_grp).cumsum().astype(float)

        # Transmission burst: abnormal volume on the current bar.
        vol_ma = volume.rolling(window=vw, min_periods=vw).mean()
        vol_ratio = (volume / vol_ma).replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(
            {
                "prevalence": prevalence,
                "down_streak": down_streak,
                "vol_ratio": vol_ratio,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EpidemicStreakParams,
    ) -> SignalFrame:
        prevalence = indicators["prevalence"]
        down_streak = indicators["down_streak"]
        vol_ratio = indicators["vol_ratio"]

        valid = prevalence.notna() & vol_ratio.notna()

        # Regime switch: low prevalence -> susceptible host pool.
        susceptible = valid & (prevalence <= float(params.susceptible_thresh))

        # Regime-dependent required streak depth (the regime modulates primitive A).
        base_streak = max(int(params.entry_streak), 1)
        extra = max(int(params.outbreak_extra), 0)
        required = np.where(susceptible.to_numpy(), base_streak, base_streak + extra)
        required = pd.Series(required, index=data.index, dtype=float)

        # Primitive A: consecutive down-streak count meets the regime threshold.
        primitive_a = valid & (down_streak >= required)
        # Primitive B: a transmission burst confirms genuine capitulation.
        primitive_b = valid & (vol_ratio >= float(params.vol_burst_mult))

        # Two-primitive AND: both must agree.
        entry_raw = (primitive_a & primitive_b).fillna(False).to_numpy()

        close_arr = data["close"].astype(float).to_numpy()
        n = len(data)
        pos = np.zeros(n, dtype=int)

        pt = float(params.profit_target)
        max_hold = max(int(params.max_hold), 1)

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if entry_raw[i]:
                    in_pos = True
                    entry_price = close_arr[i]
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                hit_target = (entry_price > 0.0) and (
                    close_arr[i] >= entry_price * (1.0 + pt)
                )
                hit_time = bars_held >= max_hold
                if hit_target or hit_time:
                    pos[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                else:
                    pos[i] = 1

        signal = pd.Series(pos, index=data.index, dtype="int64")
        size = pd.Series(1.0, index=data.index, dtype=float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = size.shift(1).fillna(1.0).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
