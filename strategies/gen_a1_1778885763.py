from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 60
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778885763"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 2 * int(params.lookback) + 20

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        lb = max(int(params.lookback), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)

        # overnight gap and its recent dispersion (signal-to-noise filter)
        gap = open_ / prev_close - 1.0
        gap_std = gap.rolling(lb).std()
        gap_z = gap / gap_std.replace(0.0, np.nan)

        # windowed drawdown-depth series
        roll_peak = close.rolling(lb).max()
        dd = close / roll_peak - 1.0
        dd_min = dd.rolling(lb).min()
        dd_max = dd.rolling(lb).max()

        # prior bar sits at a fresh extreme of the drawdown series
        fresh_deep_prev = dd.shift(1) <= dd_min.shift(1)
        fresh_peak_prev = dd.shift(1) >= dd_max.shift(1)

        # ATR for the fixed volatility stop
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        atr = atr.bfill().fillna(tr.median())

        long_raw = (fresh_deep_prev & (gap_z > 1.0)).astype(float)
        short_raw = (fresh_peak_prev & (gap_z < -1.0)).astype(float)

        out = pd.DataFrame(index=data.index)
        out["gap_z"] = gap_z.fillna(0.0)
        out["dd"] = dd.fillna(0.0)
        out["atr"] = atr.fillna(0.0)
        out["long_raw"] = long_raw.fillna(0.0)
        out["short_raw"] = short_raw.fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_raw = indicators["long_raw"].to_numpy() > 0.5
        short_raw = indicators["short_raw"].to_numpy() > 0.5
        k = float(params.atr_mult)

        pos = np.zeros(n, dtype=int)
        state = 0
        entry_px = 0.0
        stop_dist = 0.0
        entry_idx = -1

        for i in range(n):
            if state == 0:
                if long_raw[i]:
                    state = 1
                    entry_px = close[i]
                    stop_dist = k * atr[i]
                    entry_idx = i
                    pos[i] = 1
                elif short_raw[i]:
                    state = -1
                    entry_px = close[i]
                    stop_dist = k * atr[i]
                    entry_idx = i
                    pos[i] = -1
                else:
                    pos[i] = 0
            else:
                held = i - entry_idx
                if state == 1:
                    stop_hit = close[i] < entry_px - stop_dist
                else:
                    stop_hit = close[i] > entry_px + stop_dist
                if stop_hit or held >= 2:
                    pos[i] = 0
                    state = 0
                else:
                    pos[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(1.0, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
