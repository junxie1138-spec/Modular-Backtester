from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    down_streak: int = 3
    atr_trail_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779649480"

    _ATR_WIN = 14
    _DD_WIN = 60
    _MA_WIN = 200
    _DD_MIN = 0.02
    _BREAKEVEN_PCT = 0.005

    @classmethod
    def params_type(cls):
        return GenA1Params

    @classmethod
    def warmup_bars(cls, params: GenA1Params) -> int:
        return cls._MA_WIN + 20

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(cls._ATR_WIN, min_periods=cls._ATR_WIN).mean()

        roll_max = close.rolling(cls._DD_WIN, min_periods=cls._DD_WIN).max()
        dd_pct = (roll_max - close) / roll_max

        is_down = (close < prev_close).fillna(False).astype(int)
        reset_groups = (is_down == 0).cumsum()
        down_streak = is_down.groupby(reset_groups).cumsum()

        prev_streak = down_streak.shift(1).fillna(0)
        is_up = (close > prev_close).fillna(False)
        streak_broke = is_up & (prev_streak >= int(params.down_streak))

        dd_active = dd_pct.fillna(0.0) >= cls._DD_MIN
        ma200 = close.rolling(cls._MA_WIN, min_periods=cls._MA_WIN).mean()
        trend_ok = (close > ma200).fillna(False)

        entry = streak_broke & dd_active & trend_ok

        return pd.DataFrame(
            {
                "atr": atr,
                "dd_pct": dd_pct,
                "down_streak": down_streak,
                "ma200": ma200,
                "entry": entry.astype(int),
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=int)

        n = close.shape[0]
        raw_signal = np.zeros(n, dtype=np.int64)

        trail_mult = float(params.atr_trail_mult)
        breakeven_pct = cls._BREAKEVEN_PCT

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False

        for i in range(n):
            if in_pos:
                if low[i] <= stop:
                    in_pos = False
                    breakeven_done = False
                    raw_signal[i] = 0
                    continue

                if not breakeven_done:
                    if high[i] >= entry_price * (1.0 + breakeven_pct):
                        if entry_price > stop:
                            stop = entry_price
                        breakeven_done = True

                if breakeven_done and not np.isnan(atr[i]):
                    trail = close[i] - trail_mult * atr[i]
                    if trail > stop:
                        stop = trail

                raw_signal[i] = 1
            else:
                if entry[i] == 1 and not np.isnan(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    stop = close[i] - trail_mult * atr[i]
                    breakeven_done = False
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
