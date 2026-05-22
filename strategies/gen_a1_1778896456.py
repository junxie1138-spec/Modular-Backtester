from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    ma_len: int = 50
    z_len: int = 60
    pct_len: int = 120
    speed_len: int = 5
    speed_pct_len: int = 252
    hi_pct: float = 0.75
    lo_pct: float = 0.35
    trend_len: int = 200
    use_trend_gate: bool = True


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    """Shockwave front-speed momentum on the distance-from-MA z-score.

    The empirical percentile rank of the z-score is read as a density
    coordinate; the smoothed velocity of that rank is a kinematic front
    speed. A long is opened when the front speed enters the high band of
    its own rolling percentile distribution, and closed (signal-reversal)
    when a downstream front forms - the front speed sinking into the low
    band of the same distribution.
    """

    strategy_id = "gen_a1_1778896456"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        chain = (
            int(params.ma_len)
            + int(params.z_len)
            + int(params.pct_len)
            + int(params.speed_len)
            + int(params.speed_pct_len)
            + 5
        )
        return int(max(chain, int(params.trend_len) + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        ma_len = int(params.ma_len)
        z_len = int(params.z_len)
        pct_len = int(params.pct_len)
        speed_len = int(params.speed_len)
        speed_pct_len = int(params.speed_pct_len)
        trend_len = int(params.trend_len)

        ma = close.rolling(ma_len, min_periods=ma_len).mean()
        dist = (close - ma) / ma.replace(0.0, np.nan)

        d_mean = dist.rolling(z_len, min_periods=z_len).mean()
        d_std = dist.rolling(z_len, min_periods=z_len).std()
        z = (dist - d_mean) / d_std.replace(0.0, np.nan)

        # empirical percentile rank of the z-score (the density coordinate)
        pr = z.rolling(pct_len, min_periods=pct_len).rank(pct=True)

        # shockwave front speed: smoothed velocity of the percentile rank
        front_speed = pr.diff().rolling(speed_len, min_periods=speed_len).mean()

        # percentile-threshold twist: rank the front speed in its own history
        fs_rank = front_speed.rolling(
            speed_pct_len, min_periods=speed_pct_len
        ).rank(pct=True)

        trend_ma = close.rolling(trend_len, min_periods=trend_len).mean()

        ind["z"] = z
        ind["pr"] = pr
        ind["front_speed"] = front_speed
        ind["fs_rank"] = fs_rank
        ind["trend_ma"] = trend_ma
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        close = data["close"].to_numpy(dtype=float)
        fs_rank = indicators["fs_rank"].to_numpy(dtype=float)
        trend_ma = indicators["trend_ma"].to_numpy(dtype=float)

        n = len(df)
        sig = np.zeros(n, dtype=np.int64)

        hi = float(params.hi_pct)
        lo = float(params.lo_pct)
        gate = bool(params.use_trend_gate)

        pos = 0
        for i in range(n):
            fr = fs_rank[i]
            if not np.isfinite(fr):
                sig[i] = pos
                continue

            if pos == 0:
                entry_ok = fr >= hi
                if entry_ok and gate:
                    tm = trend_ma[i]
                    entry_ok = bool(np.isfinite(tm) and close[i] > tm)
                if entry_ok:
                    pos = 1
            else:
                # signal-reversal exit: a downstream (downward) front formed
                if fr <= lo:
                    pos = 0

            sig[i] = pos

        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
