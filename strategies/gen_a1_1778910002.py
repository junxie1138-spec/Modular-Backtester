from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicTakeoffParams:
    infection_window: int = 20
    accel_smooth: int = 3
    trend_ma: int = 50
    infection_floor: float = 0.30
    inflection_cap: float = 0.55
    confirm_bars: int = 2
    profit_target: float = 0.03
    time_stop: int = 2


class GeneratedStrategy(BaseStrategy[EpidemicTakeoffParams]):
    """Epidemic take-off momentum.

    Models up-move participation as a susceptible-infected contagion. The
    rolling fraction of up-closes is the infection level I_t; its rate of
    change is the epidemic growth rate and the second difference is the
    curvature of the SI curve. Entries fire only in the pre-inflection
    take-off phase: I below an inflection cap (susceptible pool abundant),
    rising, and accelerating. A two-bar confirmation gates the entry.
    Exits on a profit target or a time-stop, whichever comes first.
    """

    strategy_id = "gen_a1_1778910002"

    @classmethod
    def params_type(cls) -> type[EpidemicTakeoffParams]:
        return EpidemicTakeoffParams

    @staticmethod
    def warmup_bars(params: EpidemicTakeoffParams) -> int:
        return int(
            max(
                int(params.trend_ma),
                int(params.infection_window)
                + int(params.accel_smooth)
                + int(params.confirm_bars)
                + 5,
            )
        ) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: EpidemicTakeoffParams) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        # an up-close is a transmission event in the contagion
        up = (close > close.shift(1)).astype(float)

        # infection level I_t: fraction of the last W bars that are infected
        w = max(int(params.infection_window), 2)
        infection = up.rolling(w, min_periods=w).mean()

        # smooth the epidemic curve before measuring its curvature
        s = max(int(params.accel_smooth), 1)
        inf_smooth = infection.rolling(s, min_periods=s).mean()

        # epidemic growth rate and its acceleration (curvature of the SI curve)
        growth = inf_smooth.diff()
        accel = growth.diff()

        # susceptible pool: fuel still available for the contagion to spread
        susceptible = 1.0 - inf_smooth

        ma_len = max(int(params.trend_ma), 2)
        trend = close.rolling(ma_len, min_periods=ma_len).mean()

        ind["infection"] = inf_smooth
        ind["growth"] = growth
        ind["accel"] = accel
        ind["susceptible"] = susceptible
        ind["trend"] = trend

        # pre-inflection take-off condition for a single bar:
        #   - epidemic still early: I below the inflection cap, above a floor
        #   - contagion spreading: growth > 0
        #   - contagion accelerating: accel > 0 (pre-inflection regime)
        #   - price in an uptrend (long-only momentum gate)
        cond = (
            (inf_smooth < float(params.inflection_cap))
            & (inf_smooth > float(params.infection_floor))
            & (growth > 0.0)
            & (accel > 0.0)
            & (close > trend)
        )
        cond = cond.fillna(False)

        # two-bar confirmation twist: the take-off must persist
        k = max(int(params.confirm_bars), 1)
        entry = cond.copy()
        for j in range(1, k):
            entry = entry & cond.shift(j).fillna(False)
        ind["entry_signal"] = entry.fillna(False).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EpidemicTakeoffParams,
    ) -> SignalFrame:
        close = np.nan_to_num(
            data["close"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        entry = indicators["entry_signal"].to_numpy(dtype=float) > 0.5
        n = len(close)

        pos = np.zeros(n, dtype=np.int64)
        pt = float(params.profit_target)
        max_hold = max(int(params.time_stop), 1)

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if entry[i] and close[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                hit_pt = entry_price > 0.0 and close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
