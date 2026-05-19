from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    ma_len: int = 50
    z_len: int = 50
    z_arm: float = -1.5
    z_high: float = 0.6
    speed_len: int = 3
    speed_smooth: int = 2
    vol_len: int = 20
    target_vol: float = 0.15
    strength_gain: float = 0.5


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    """Long the propagating recovery shockwave of a deep MA-distance z-score.

    A deep negative distance-from-MA z-score arms the strategy. The recovery
    front is the smoothed velocity of the z-score; when that front carries the
    z-score back up past the arming level the trade enters. The position is
    held until the entry state flips - either equilibrium is restored (z-score
    reaches z_high) or the front reverses (negative shock speed). Size is the
    product of a realized-volatility target scalar and the measured strength of
    the shockwave (discount depth plus front speed).
    """

    strategy_id = "gen_a2_1779153071"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        p = params
        a = p.ma_len + p.z_len + p.speed_len + p.speed_smooth + 2
        b = p.vol_len + 2
        return int(max(a, b))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        p = params
        close = data["close"]
        out = pd.DataFrame(index=data.index)

        sma = close.rolling(p.ma_len, min_periods=p.ma_len).mean()
        dist = close - sma
        dist_mean = dist.rolling(p.z_len, min_periods=p.z_len).mean()
        dist_std = dist.rolling(p.z_len, min_periods=p.z_len).std(ddof=0)
        dist_std = dist_std.replace(0.0, np.nan)
        z = (dist - dist_mean) / dist_std

        raw_speed = z.diff(p.speed_len)
        shock_speed = raw_speed.rolling(
            p.speed_smooth, min_periods=p.speed_smooth
        ).mean()

        ret = close.pct_change()
        rv = ret.rolling(p.vol_len, min_periods=p.vol_len).std(ddof=0) * np.sqrt(252.0)
        rv = rv.replace(0.0, np.nan)
        vol_scalar = (p.target_vol / rv).clip(lower=0.2, upper=1.5)

        out["z"] = z
        out["shock_speed"] = shock_speed
        out["vol_scalar"] = vol_scalar
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        p = params
        idx = data.index
        n = len(idx)

        z = indicators["z"].to_numpy(dtype=np.float64)
        v = indicators["shock_speed"].to_numpy(dtype=np.float64)
        vol_scalar = indicators["vol_scalar"].to_numpy(dtype=np.float64)

        raw_sig = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_pos = False
        armed = False
        z_min = 0.0
        cur_size = 1.0

        for i in range(n):
            zi = z[i]
            vi = v[i]
            vsi = vol_scalar[i]

            if not (np.isfinite(zi) and np.isfinite(vi) and np.isfinite(vsi)):
                raw_sig[i] = 1 if in_pos else 0
                size[i] = cur_size if in_pos else 1.0
                continue

            if in_pos:
                # Signal-reversal exit: the entry state (a rising recovery front
                # below equilibrium) has flipped - equilibrium restored or the
                # shockwave front reversed.
                if zi >= p.z_high or vi < 0.0:
                    in_pos = False
                    raw_sig[i] = 0
                    size[i] = cur_size
                else:
                    raw_sig[i] = 1
                    size[i] = cur_size
                continue

            # Flat: maintain the arming state on deep discount.
            if zi < p.z_arm:
                if not armed:
                    armed = True
                    z_min = zi
                else:
                    z_min = min(z_min, zi)

            # Entry: the recovery front has carried the z-score back up past
            # the arming level with a positive (advancing) shock speed.
            if armed and zi > p.z_arm and vi > 0.0:
                depth_mag = min(-z_min, 3.0)
                speed_mag = min(vi, 3.0)
                strength = 0.5 + p.strength_gain * (depth_mag / 1.5 + speed_mag)
                strength = float(np.clip(strength, 0.5, 2.0))
                cur_size = float(np.clip(vsi * strength, 0.1, 2.0))
                in_pos = True
                armed = False
                z_min = 0.0
                raw_sig[i] = 1
                size[i] = cur_size
            else:
                raw_sig[i] = 0
                size[i] = 1.0

        df = pd.DataFrame(index=idx)
        df["signal"] = raw_sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
