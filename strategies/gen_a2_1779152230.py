from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# --- fixed structural constants (kept out of the param class to honor the
# --- <=2 tunable params twist) ---
MA_WIN = 20        # moving average window for the distance measure
DIST_WIN = 20      # rolling std window for standardizing MA distance
GAP_WIN = 20       # rolling std window for standardizing the overnight gap
GAP_Z_MIN = 1.0    # gap must be at least this many std devs to count as a gap
TIME_STOP = 5      # hard time-stop in bars (3-5 day holding horizon)


@dataclass(slots=True)
class GapDisplacementParams:
    z_band: float = 1.5      # MA-distance z-score extreme required to fade
    profit_pct: float = 0.03  # profit target as a fraction of entry price


class GeneratedStrategy(BaseStrategy[GapDisplacementParams]):
    strategy_id = "gen_a2_1779152230"

    @classmethod
    def params_type(cls):
        return GapDisplacementParams

    @staticmethod
    def warmup_bars(params: GapDisplacementParams) -> int:
        # MA distance needs MA_WIN bars, then DIST_WIN bars of that distance,
        # plus one bar for the prev-close shift used by the gap.
        return MA_WIN + DIST_WIN + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapDisplacementParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        prev_close = close.shift(1)

        # Overnight gap, standardized into a z-score.
        gap_pct = (open_ - prev_close) / prev_close
        gap_std = gap_pct.rolling(GAP_WIN, min_periods=GAP_WIN).std()
        gap_z = gap_pct / gap_std.replace(0.0, np.nan)

        # Distance of close from its moving average, standardized into a z-score.
        sma = close.rolling(MA_WIN, min_periods=MA_WIN).mean()
        dist = close - sma
        dist_std = dist.rolling(DIST_WIN, min_periods=DIST_WIN).std()
        zscore = dist / dist_std.replace(0.0, np.nan)

        out = pd.DataFrame(index=data.index)
        out["gap_z"] = gap_z.replace([np.inf, -np.inf], np.nan)
        out["zscore"] = zscore.replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDisplacementParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap_z = indicators["gap_z"].to_numpy(dtype=float)
        zscore = indicators["zscore"].to_numpy(dtype=float)

        z_band = float(params.z_band)
        profit_pct = float(params.profit_pct)
        rearm_level = z_band * 0.5  # hysteresis: price must recover halfway back

        pos = np.zeros(n, dtype=int)
        state = 0
        entry_price = 0.0
        entry_i = 0
        # Hysteresis latches: a side stays disarmed after a trade until the
        # z-score has crossed back inside its neutral re-arm level.
        armed_long = True
        armed_short = True

        for i in range(n):
            gz = gap_z[i]
            zs = zscore[i]

            if not np.isfinite(gz) or not np.isfinite(zs):
                # Indeterminate inputs: hold whatever position we have.
                pos[i] = state
                continue

            if state == 0:
                # Re-arm a side once price has recovered toward the MA.
                if not armed_long and zs > -rearm_level:
                    armed_long = True
                if not armed_short and zs < rearm_level:
                    armed_short = True

                # Down-gap that drove the close into a low z-score extreme.
                if armed_long and gz <= -GAP_Z_MIN and zs <= -z_band:
                    state = 1
                    entry_price = close[i]
                    entry_i = i
                    armed_long = False
                # Up-gap that drove the close into a high z-score extreme.
                elif armed_short and gz >= GAP_Z_MIN and zs >= z_band:
                    state = -1
                    entry_price = close[i]
                    entry_i = i
                    armed_short = False
            else:
                held = i - entry_i
                exit_now = False
                if state == 1:
                    if close[i] >= entry_price * (1.0 + profit_pct):
                        exit_now = True
                    elif held >= TIME_STOP:
                        exit_now = True
                else:
                    if close[i] <= entry_price * (1.0 - profit_pct):
                        exit_now = True
                    elif held >= TIME_STOP:
                        exit_now = True
                if exit_now:
                    state = 0

            pos[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
