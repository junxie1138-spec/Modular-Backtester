from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CrowdGatedZParams:
    ma_len: int = 50
    z_window: int = 60
    z_entry: float = 1.0
    crowd_window: int = 20
    crowd_capacity: int = 8
    hold_bars: int = 4
    regime_ma: int = 200
    base_size: float = 1.0
    size_capacity_scale: float = 0.5


class GeneratedStrategy(BaseStrategy[CrowdGatedZParams]):
    strategy_id = "gen_a1_1778889061"

    @classmethod
    def params_type(cls) -> type[CrowdGatedZParams]:
        return CrowdGatedZParams

    @staticmethod
    def warmup_bars(params: CrowdGatedZParams) -> int:
        return int(max(params.ma_len + params.z_window,
                       params.regime_ma,
                       params.crowd_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CrowdGatedZParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        # Distance-from-MA z-score: how far price has extended from its mean.
        sma = close.rolling(params.ma_len).mean()
        dist = close - sma
        dist_std = dist.rolling(params.z_window).std()
        z = dist / dist_std.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)

        # Velocity of the z-score: confirms the move is still accelerating.
        dz = z.diff()

        # 200-day regime filter (the hard twist).
        regime = close.rolling(params.regime_ma).mean()

        # Crowding queue: count of recent bars already in the extended state.
        # A full queue means the move is stale / overcrowded.
        above = (z.abs() > params.z_entry).astype(float)
        crowd = above.rolling(params.crowd_window).sum()

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["dz"] = dz
        out["regime"] = regime
        out["crowd"] = crowd
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: CrowdGatedZParams) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        dz = indicators["dz"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)
        crowd = indicators["crowd"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, float(params.base_size), dtype=float)

        position = 0
        entry_bar = -1
        entry_size = float(params.base_size)
        cap = max(1, int(params.crowd_capacity))
        hold = max(1, int(params.hold_bars))

        for i in range(n):
            # --- Fixed-bar exit: flatten exactly `hold` bars after entry. ---
            if position != 0:
                if i - entry_bar >= hold:
                    position = 0
                    entry_bar = -1
                else:
                    signal[i] = position
                    size[i] = entry_size

            # --- Entry: only when flat. ---
            if position == 0:
                zi = z[i]
                dzi = dz[i]
                ri = regime[i]
                ci = crowd[i]
                if (np.isnan(zi) or np.isnan(dzi)
                        or np.isnan(ri) or np.isnan(ci)):
                    continue
                # Capacity limit: skip crowded / stale moves.
                if ci > params.crowd_capacity:
                    continue

                headroom = max(0.0, cap - ci) / cap
                pos_size = float(params.base_size) * (
                    1.0 + params.size_capacity_scale * headroom)
                if not (pos_size > 0.0):
                    pos_size = float(params.base_size)

                long_cond = (zi > params.z_entry and dzi > 0.0
                             and close[i] > ri)
                short_cond = (zi < -params.z_entry and dzi < 0.0
                              and close[i] < ri)

                if long_cond:
                    position = 1
                    entry_bar = i
                    entry_size = pos_size
                    signal[i] = 1
                    size[i] = pos_size
                elif short_cond:
                    position = -1
                    entry_bar = i
                    entry_size = pos_size
                    signal[i] = -1
                    size[i] = pos_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        # Mandatory one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(float(params.base_size))
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
