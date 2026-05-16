from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel: int = 40
    ma_len: int = 200
    atr_len: int = 14
    vol_baseline: int = 60
    vol_expansion: float = 1.05
    vol_accel: int = 5
    k_stop: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778894984"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        p = params
        lookbacks = [
            p.ma_len,
            p.channel + 1,
            p.atr_len + 1,
            p.atr_len + p.vol_baseline,
            p.vol_accel + 1,
        ]
        return int(max(lookbacks)) + 5

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        p = params
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(p.atr_len, min_periods=p.atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(p.ma_len, min_periods=p.ma_len).mean()
        out["channel_high"] = high.rolling(p.channel, min_periods=p.channel).max()
        out["channel_low"] = low.rolling(p.channel, min_periods=p.channel).min()
        out["atr"] = atr
        out["atr_baseline"] = atr.rolling(
            p.vol_baseline, min_periods=p.vol_baseline
        ).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        p = params
        close = data["close"]

        atr = indicators["atr"]
        atr_baseline = indicators["atr_baseline"]
        ma200 = indicators["ma200"]
        ch_high_prev = indicators["channel_high"].shift(1)
        ch_low_prev = indicators["channel_low"].shift(1)

        # Volatility is igniting: above its multi-week baseline AND accelerating.
        expanding = (atr > atr_baseline * p.vol_expansion) & (
            atr > atr.shift(p.vol_accel)
        )

        bull = close > ma200
        bear = close < ma200

        long_raw = (close > ch_high_prev) & expanding & bull
        short_raw = (close < ch_low_prev) & expanding & bear

        # NaN comparisons during warmup evaluate False -> no spurious entries.
        raw = np.where(
            long_raw.fillna(False).to_numpy(),
            1,
            np.where(short_raw.fillna(False).to_numpy(), -1, 0),
        ).astype(int)

        close_arr = close.to_numpy(dtype=float)
        atr_arr = atr.to_numpy(dtype=float)
        n = len(close_arr)

        pos = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                s = int(raw[i])
                a = atr_arr[i]
                if s != 0 and np.isfinite(a) and a > 0.0:
                    position = s
                    entry_price = close_arr[i]
                    entry_atr = a
                    bars_held = 0
                pos[i] = position
                continue

            bars_held += 1
            exit_now = False

            if position == 1:
                stop = entry_price - p.k_stop * entry_atr
                if close_arr[i] <= stop:
                    exit_now = True
            else:
                stop = entry_price + p.k_stop * entry_atr
                if close_arr[i] >= stop:
                    exit_now = True

            if bars_held >= p.max_hold:
                exit_now = True

            if exit_now:
                position = 0
                entry_price = 0.0
                entry_atr = 0.0
                bars_held = 0

            pos[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
