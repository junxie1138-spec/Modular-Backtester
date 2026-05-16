from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    win: int = 4
    floor_density: float = 0.10
    speed_thresh: float = 0.0
    min_flow: float = 0.0
    size_gain: float = 50.0


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_a1_1778904546"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        # two adjacent windows of returns + 1 bar consumed by pct_change
        return 2 * max(2, int(params.win)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        win = max(2, int(params.win))
        floor_density = float(params.floor_density)

        close = data["close"].astype(float)
        ret = close.pct_change()

        # Downstream (recent) traffic state: flow = net return throughput,
        # density = occupancy fraction of up-moving bars.
        flow_r = ret.rolling(win).mean()
        density_r = (ret > 0).astype(float).rolling(win).mean()

        # Upstream (prior) traffic state: the window immediately before it.
        flow_o = flow_r.shift(win)
        density_o = density_r.shift(win)

        # Rankine-Hugoniot jump: shockwave speed across the regime interface.
        # delta-density floored so the ratio stays finite (a contact
        # discontinuity with little density change yields a sharp, fast front).
        d_density = (density_r - density_o).abs() + floor_density
        wave_speed = (flow_r - flow_o) / d_density

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["flow_r"] = flow_r
        out["wave_speed"] = wave_speed
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        wave_speed = indicators["wave_speed"]
        flow_r = indicators["flow_r"]

        speed_thresh = float(params.speed_thresh)
        min_flow = float(params.min_flow)

        # Entry condition: a forward-propagating positive trend shockwave.
        # Signal-reversal exit - long-only position is held while this same
        # condition holds and is driven to 0 the moment the condition flips.
        cond = (wave_speed > speed_thresh) & (flow_r > min_flow)
        cond = cond.fillna(False)
        raw_signal = cond.astype(int)

        # Conviction-scaled size, bounded and strictly positive.
        size = 1.0 + 0.5 * np.tanh(wave_speed.fillna(0.0) * float(params.size_gain))
        size = size.clip(lower=0.5, upper=1.5)

        df = pd.DataFrame(index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
