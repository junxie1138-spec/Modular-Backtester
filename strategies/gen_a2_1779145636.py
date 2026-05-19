from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionShockParams:
    short_win: int = 5
    long_win: int = 60
    compression_thresh: float = 0.70
    atr_win: int = 14
    atr_k: float = 2.0
    max_hold: int = 10
    size_floor: float = 0.45
    size_cap: float = 1.40


class GeneratedStrategy(BaseStrategy[CompressionShockParams]):
    strategy_id = "gen_a2_1779145636"

    @classmethod
    def params_type(cls):
        return CompressionShockParams

    @staticmethod
    def warmup_bars(params: CompressionShockParams) -> int:
        return int(max(params.long_win, params.atr_win) + params.short_win + 5)

    def indicators(self, data: pd.DataFrame, params: CompressionShockParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()

        short_std = ret.rolling(params.short_win).std()
        long_std = ret.rolling(params.long_win).std()
        ratio = short_std / long_std.replace(0.0, np.nan)

        # accumulated drift over the compression window
        drift = close.pct_change(params.short_win)
        # expected scale of a short_win-bar return sum under the baseline
        drift_scale = long_std * np.sqrt(float(params.short_win))
        drift_z = (drift / drift_scale.replace(0.0, np.nan)).abs()
        drift_z = drift_z.clip(lower=0.0, upper=1.0)

        comp_depth = ((params.compression_thresh - ratio) / params.compression_thresh)
        comp_depth = comp_depth.clip(lower=0.0, upper=1.0)

        # signal-strength score blends compression depth and drift conviction
        score = 0.5 * comp_depth + 0.5 * drift_z
        size_score = params.size_floor + score * (params.size_cap - params.size_floor)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_win).mean()

        ind = pd.DataFrame(index=data.index)
        ind["ret"] = ret
        ind["ratio"] = ratio
        ind["drift"] = drift
        ind["comp_depth"] = comp_depth
        ind["size_score"] = size_score
        ind["atr"] = atr
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionShockParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ratio = indicators["ratio"].to_numpy(dtype=float)
        drift = indicators["drift"].to_numpy(dtype=float)
        size_score = np.nan_to_num(
            indicators["size_score"].to_numpy(dtype=float),
            nan=params.size_floor,
        )
        atr = indicators["atr"].to_numpy(dtype=float)

        # armed when return dispersion is compressed below baseline fraction
        armed = np.isfinite(ratio) & (ratio < params.compression_thresh)
        direction = np.sign(np.nan_to_num(drift, nan=0.0)).astype(int)

        pos = np.zeros(n, dtype=int)
        sizes = np.full(n, params.size_floor, dtype=float)

        state = 0
        hwm = 0.0
        lwm = 0.0
        bars_held = 0
        entry_size = params.size_floor

        for i in range(n):
            c = close[i]
            if state == 0:
                if armed[i] and direction[i] != 0:
                    state = int(direction[i])
                    hwm = c
                    lwm = c
                    bars_held = 0
                    entry_size = float(size_score[i])
                    pos[i] = state
                    sizes[i] = entry_size
                continue

            bars_held += 1
            if state == 1 and c > hwm:
                hwm = c
            if state == -1 and c < lwm:
                lwm = c

            atr_i = atr[i]
            exit_now = False
            if np.isfinite(atr_i):
                if state == 1 and c <= hwm - params.atr_k * atr_i:
                    exit_now = True
                elif state == -1 and c >= lwm + params.atr_k * atr_i:
                    exit_now = True
            if bars_held >= params.max_hold:
                exit_now = True

            if exit_now:
                state = 0
                pos[i] = 0
                sizes[i] = params.size_floor
            else:
                pos[i] = state
                sizes[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)
        size_series = pd.Series(sizes, index=data.index).shift(1)
        size_series = size_series.fillna(params.size_floor)
        df["size"] = size_series.clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
