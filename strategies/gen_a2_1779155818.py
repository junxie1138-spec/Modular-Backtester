from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA2Params:
    atr_period: int = 14
    be_trigger_pct: float = 0.03
    atr_mult: float = 2.5
    vol_pct_window: int = 60
    vol_pct_threshold: float = 0.70
    seas_rank_window: int = 252
    seas_pct_threshold: float = 0.60


class GeneratedStrategy(BaseStrategy[GenA2Params]):
    strategy_id = "gen_a2_1779155818"

    @classmethod
    def params_type(cls) -> type[GenA2Params]:
        return GenA2Params

    @staticmethod
    def warmup_bars(params: GenA2Params) -> int:
        return int(max(params.atr_period, params.vol_pct_window,
                       params.seas_rank_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA2Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(int(params.atr_period), min_periods=1).mean()
        atr = atr.replace(0.0, np.nan).ffill().bfill()
        atr = atr.fillna((high - low).abs()).fillna(0.0)

        ret = np.log(close / prev_close)
        ret_filled = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        month = pd.Series(data.index.month, index=data.index)
        grp = ret_filled.groupby(month)
        prior_sum = grp.cumsum() - ret_filled
        prior_count = grp.cumcount()
        seasonal_mean = prior_sum / prior_count.where(prior_count > 0, np.nan)
        seasonal_mean = seasonal_mean.fillna(0.0)

        rw = max(int(params.seas_rank_window), 2)
        seas_rank = seasonal_mean.rolling(rw, min_periods=max(rw // 2, 2)).rank(pct=True)

        vw = max(int(params.vol_pct_window), 2)
        vol_rank = volume.rolling(vw, min_periods=max(vw // 2, 2)).rank(pct=True)

        up_move = (close > prev_close).astype(float)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["seasonal_mean"] = seasonal_mean
        out["seas_rank"] = seas_rank.fillna(0.0)
        out["vol_rank"] = vol_rank.fillna(0.0)
        out["up_move"] = up_move.fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GenA2Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        seas_rank = indicators["seas_rank"].to_numpy(dtype=float)
        vol_rank = indicators["vol_rank"].to_numpy(dtype=float)
        up_move = indicators["up_move"].to_numpy(dtype=float)

        n = len(close)
        entry_ok = ((seas_rank >= float(params.seas_pct_threshold)) &
                    (vol_rank >= float(params.vol_pct_threshold)) &
                    (up_move > 0.5))

        raw = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        stop = 0.0
        highest = 0.0
        be_armed = False
        k = float(params.atr_mult)
        be = float(params.be_trigger_pct)

        for i in range(n):
            if position == 0:
                if bool(entry_ok[i]) and atr[i] > 0.0 and np.isfinite(close[i]):
                    position = 1
                    entry_price = close[i]
                    highest = high[i]
                    stop = entry_price - k * atr[i]
                    be_armed = False
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                if high[i] > highest:
                    highest = high[i]
                if (not be_armed) and high[i] >= entry_price * (1.0 + be):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price
                if be_armed and atr[i] > 0.0:
                    trail = highest - k * atr[i]
                    if trail > stop:
                        stop = trail
                if low[i] <= stop:
                    position = 0
                    be_armed = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
