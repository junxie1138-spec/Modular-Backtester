from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideNodeParams:
    channel_len: int = 40
    amp_fast: int = 5
    amp_slow: int = 30
    compression_thr: float = 0.75
    floor_thr: float = 0.20
    ceil_thr: float = 0.80
    profit_target: float = 0.06
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[TideNodeParams]):
    strategy_id = "gen_a2_1779146345"

    @classmethod
    def params_type(cls) -> type[TideNodeParams]:
        return TideNodeParams

    @staticmethod
    def warmup_bars(params: TideNodeParams) -> int:
        return int(max(params.channel_len, params.amp_slow)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: TideNodeParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Primitive 1: high-low range amplitude dynamics.
        hl_range = (high - low).clip(lower=0.0)
        amp_fast = hl_range.rolling(
            int(params.amp_fast), min_periods=int(params.amp_fast)
        ).mean()
        amp_slow = hl_range.rolling(
            int(params.amp_slow), min_periods=int(params.amp_slow)
        ).mean()
        range_ratio = amp_fast / amp_slow.replace(0.0, np.nan)

        # Primitive 2: relative position inside the rolling high-low channel.
        chan_hi = high.rolling(
            int(params.channel_len), min_periods=int(params.channel_len)
        ).max()
        chan_lo = low.rolling(
            int(params.channel_len), min_periods=int(params.channel_len)
        ).min()
        span = (chan_hi - chan_lo).replace(0.0, np.nan)
        rel_pos = (close - chan_lo) / span

        out = pd.DataFrame(index=data.index)
        out["rel_pos"] = rel_pos
        out["range_ratio"] = range_ratio
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideNodeParams,
    ) -> SignalFrame:
        n = len(data)
        rel_pos = indicators["rel_pos"].to_numpy(dtype=float)
        range_ratio = indicators["range_ratio"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        # NaN-safe gates: warmup NaNs resolve to non-triggering values.
        compressed = np.nan_to_num(range_ratio, nan=np.inf) < float(
            params.compression_thr
        )
        at_floor = np.nan_to_num(rel_pos, nan=0.5) < float(params.floor_thr)
        at_ceil = np.nan_to_num(rel_pos, nan=0.5) > float(params.ceil_thr)

        # Two-primitive AND: range collapse AND channel extreme must agree.
        long_entry = compressed & at_floor
        short_entry = compressed & at_ceil

        pt = float(params.profit_target)
        ts = int(params.time_stop)

        pos = np.zeros(n, dtype=int)
        state = 0
        entry_price = 0.0
        entry_idx = 0
        for i in range(n):
            if state == 0:
                if long_entry[i]:
                    state = 1
                    entry_price = close[i]
                    entry_idx = i
                elif short_entry[i]:
                    state = -1
                    entry_price = close[i]
                    entry_idx = i
            else:
                held = i - entry_idx
                if state == 1:
                    gain = (
                        close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                    )
                    if gain >= pt or held >= ts:
                        state = 0
                else:
                    gain = (
                        entry_price / close[i] - 1.0 if close[i] > 0.0 else 0.0
                    )
                    if gain >= pt or held >= ts:
                        state = 0
            pos[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
