from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapJumpRetentionParams:
    vol_window: int = 20
    atr_window: int = 14
    jump_z: float = 2.0
    retain_k: float = 0.5
    trail_k: float = 2.0
    max_hold: int = 5
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[GapJumpRetentionParams]):
    """Close-to-close return-jump strategy with plastic-retention confirmation.

    A bar whose close-to-close return is >= jump_z return-standard-deviations is
    treated as a discontinuity in the close path. The strategy then waits two
    bars: if both subsequent closes retain the jumped level (stay at or above
    jump_close - retain_k * ATR), the deformation is 'plastic' and a long is
    opened. Exit is a rolling-high ATR trailing stop that only ratchets up.
    """

    strategy_id = "gen_a1_1778894834"

    @classmethod
    def params_type(cls) -> type[GapJumpRetentionParams]:
        return GapJumpRetentionParams

    def warmup_bars(self, params: GapJumpRetentionParams) -> int:
        return int(max(params.vol_window, params.atr_window)) + 2

    def indicators(self, data: pd.DataFrame, params: GapJumpRetentionParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        ret_std = ret.rolling(params.vol_window, min_periods=params.vol_window).std()
        ret_std = ret_std.replace(0.0, np.nan)
        ret_z = (ret / ret_std).replace([np.inf, -np.inf], np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["ret_z"] = ret_z
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapJumpRetentionParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        ret_z = indicators["ret_z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        # state: 0 = scanning, 1 = confirming retention, 2 = in position
        state = 0
        jump_close = 0.0
        jump_atr = 0.0
        jump_strength = 1.0
        confirms = 0
        hwm = 0.0
        hold = 0
        entry_size = 1.0

        for i in range(n):
            a = atr[i]
            rz = ret_z[i]
            valid = (not np.isnan(a)) and a > 0.0

            if state == 0:
                if valid and (not np.isnan(rz)) and rz >= params.jump_z and ret[i] > 0.0:
                    state = 1
                    jump_close = close[i]
                    jump_atr = a
                    jump_strength = rz
                    confirms = 0

            elif state == 1:
                if not valid:
                    continue
                floor = jump_close - params.retain_k * jump_atr
                if close[i] >= floor:
                    confirms += 1
                    if confirms >= 2:
                        state = 2
                        signal[i] = 1
                        hwm = close[i]
                        hold = 0
                        entry_size = min(
                            max(jump_strength / params.jump_z, 1.0),
                            params.size_cap,
                        )
                        size[i] = entry_size
                else:
                    # elastic snap-back: jump not retained, abandon
                    state = 0

            elif state == 2:
                if not valid:
                    hold += 1
                    if hold >= params.max_hold:
                        state = 0
                        signal[i] = 0
                    else:
                        signal[i] = 1
                        size[i] = entry_size
                    continue
                hold += 1
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.trail_k * a
                if close[i] < stop or hold >= params.max_hold:
                    state = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
