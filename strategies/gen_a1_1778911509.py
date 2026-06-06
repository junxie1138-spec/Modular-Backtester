from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringReleaseParams:
    ma_period: int = 20
    z_period: int = 20
    z_smooth: int = 3
    z_entry: float = -1.5
    dd_period: int = 60
    dd_min: float = 0.03
    regime_period: int = 200
    atr_period: int = 14
    init_stop_atr_mult: float = 2.5
    trail_atr_mult: float = 3.0
    breakeven_pct: float = 0.03
    max_hold: int = 12
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[SpringReleaseParams]):
    strategy_id = "gen_a1_1778911509"

    @classmethod
    def params_type(cls) -> type[SpringReleaseParams]:
        return SpringReleaseParams

    @staticmethod
    def warmup_bars(params: SpringReleaseParams) -> int:
        z_chain = params.ma_period + params.z_period + params.z_smooth + 2
        return int(max(
            params.regime_period,
            z_chain,
            params.dd_period,
            params.atr_period + 1,
        )) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: SpringReleaseParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        distance = close - ma
        dist_mean = distance.rolling(params.z_period, min_periods=params.z_period).mean()
        dist_std = distance.rolling(params.z_period, min_periods=params.z_period).std()
        z = (distance - dist_mean) / dist_std.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        z_smooth = z.rolling(params.z_smooth, min_periods=params.z_smooth).mean()
        z_vel = z_smooth.diff()

        roll_high = close.rolling(params.dd_period, min_periods=params.dd_period).max()
        drawdown = (close - roll_high) / roll_high.replace(0.0, np.nan)

        ma200 = close.rolling(params.regime_period, min_periods=params.regime_period).mean()

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["z"] = z
        out["z_smooth"] = z_smooth
        out["z_vel"] = z_vel
        out["drawdown"] = drawdown
        out["ma200"] = ma200
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringReleaseParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        z_smooth = indicators["z_smooth"].to_numpy(dtype=float)
        z_vel = indicators["z_vel"].to_numpy(dtype=float)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        pos = np.zeros(n, dtype=np.int64)

        in_pos = False
        entry_price = 0.0
        entry_idx = 0
        stop = 0.0
        be_done = False
        hi_since = 0.0

        for i in range(n):
            c = close[i]
            a = atr[i]

            if not in_pos:
                entry_ok = (
                    np.isfinite(z_smooth[i])
                    and np.isfinite(z_vel[i])
                    and np.isfinite(drawdown[i])
                    and np.isfinite(ma200[i])
                    and np.isfinite(a)
                    and a > 0.0
                    and c > ma200[i]
                    and drawdown[i] <= -params.dd_min
                    and z_smooth[i] <= params.z_entry
                    and z_vel[i] > 0.0
                )
                if entry_ok:
                    in_pos = True
                    entry_price = c
                    entry_idx = i
                    stop = c - params.init_stop_atr_mult * a
                    be_done = False
                    hi_since = c
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                if c > hi_since:
                    hi_since = c
                if (not be_done) and c >= entry_price * (1.0 + params.breakeven_pct):
                    if entry_price > stop:
                        stop = entry_price
                    be_done = True
                if be_done and np.isfinite(a):
                    trail = hi_since - params.trail_atr_mult * a
                    if trail > stop:
                        stop = trail
                held = i - entry_idx
                if c <= stop or held >= params.max_hold:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = float(params.base_size)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
