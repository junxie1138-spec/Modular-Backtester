from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback_weeks: int = 26
    streak_thresh: int = 3
    atr_period: int = 14
    atr_k: float = 1.5
    max_streak: int = 8


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778888159"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.lookback_weeks * 5 + 10, params.atr_period + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        # Rolling median return for each weekday over past lookback_weeks same-weekday bars
        parts = []
        for dow in range(5):
            mask = data.index.dayofweek == dow
            dow_idx = data.index[mask]
            if mask.sum() < params.lookback_weeks + 2:
                parts.append(pd.Series(np.nan, index=dow_idx, dtype=float))
                continue
            dow_ret = ret.loc[mask]
            baseline = dow_ret.shift(1).rolling(params.lookback_weeks).median()
            parts.append(baseline)
        seasonal_baseline = pd.concat(parts).sort_index().reindex(data.index)

        excess = ret - seasonal_baseline

        # Beat direction: +1 if above weekday baseline, -1 if below, 0 if invalid
        beat = pd.Series(
            np.sign(np.where(np.isfinite(excess), excess, 0.0)).astype(int),
            index=data.index,
        )

        # Vectorised consecutive-streak count for same non-zero beat direction
        direction_change = (beat != beat.shift(1)) | (beat == 0)
        groups = direction_change.cumsum()
        streak_raw = beat.groupby(groups).cumcount() + 1
        streak = streak_raw.where(beat != 0, other=0).astype(float)

        # Zero-out / NaN warmup bars
        valid = seasonal_baseline.notna() & atr.notna()
        beat = beat.where(valid, other=0)
        streak = streak.where(valid, other=np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["beat"] = beat
        ind["streak"] = streak
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        beat = indicators["beat"].values
        streak = indicators["streak"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.5, dtype=float)

        in_trade = False
        trade_dir = 0
        stop_price = 0.0
        entry_size = 0.5

        for i in range(n):
            s = streak[i]
            a = atr[i]

            if np.isnan(s) or np.isnan(a):
                signal[i] = 0
                size[i] = 0.5
                continue

            if in_trade:
                if trade_dir == 1 and close[i] <= stop_price:
                    signal[i] = 0
                    in_trade = False
                    trade_dir = 0
                elif trade_dir == -1 and close[i] >= stop_price:
                    signal[i] = 0
                    in_trade = False
                    trade_dir = 0
                else:
                    signal[i] = trade_dir
                    size[i] = entry_size
            else:
                b = int(beat[i])
                if s >= params.streak_thresh and b != 0:
                    # Fade the exhausted seasonal streak (predator-prey reversal)
                    new_dir = -b
                    scale = min(float(s) / float(params.max_streak), 1.0)
                    entry_size = 0.3 + 0.7 * scale
                    signal[i] = new_dir
                    size[i] = entry_size
                    in_trade = True
                    trade_dir = new_dir
                    if new_dir == 1:
                        stop_price = close[i] - params.atr_k * a
                    else:
                        stop_price = close[i] + params.atr_k * a

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.5).clip(lower=0.01)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
