from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalGapParams:
    atr_len: int = 14
    vol_len: int = 20
    season_lookback: int = 16
    min_occurrences: int = 8
    gap_z_enter: float = 1.0
    seasonal_enter: float = 0.0006
    k_atr: float = 1.75
    max_hold: int = 3
    target_vol: float = 0.012
    size_floor: float = 0.3
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[SeasonalGapParams]):
    strategy_id = "gen_a1_1778889198"

    @classmethod
    def params_type(cls):
        return SeasonalGapParams

    @staticmethod
    def warmup_bars(params: SeasonalGapParams) -> int:
        return int(max(params.atr_len + 1,
                       params.vol_len + 1,
                       params.season_lookback * 5 + 10))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalGapParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        prev_close = close.shift(1)

        # Primary primitive: overnight gap (open vs prior close).
        gap = open_ / prev_close - 1.0

        # ATR for the trailing stop.
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        # Gap z-score: how unusual is today's gap vs its recent distribution.
        gap_mean = gap.rolling(params.vol_len, min_periods=params.vol_len).mean()
        gap_std = gap.rolling(params.vol_len, min_periods=params.vol_len).std()
        gap_z = (gap - gap_mean) / gap_std.replace(0.0, np.nan)

        # Seasonal primitive: per-weekday rolling climatology of the gap.
        # shift(1) inside the group keeps today's gap out of its own mean.
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        seasonal_gap = gap.groupby(weekday).transform(
            lambda s: s.shift(1).rolling(params.season_lookback,
                                         min_periods=params.min_occurrences).mean()
        )

        # Realized vol for inverse-vol position sizing.
        ret = close.pct_change()
        realized_vol = ret.rolling(params.vol_len, min_periods=params.vol_len).std()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_z"] = gap_z
        out["seasonal_gap"] = seasonal_gap
        out["atr"] = atr
        out["realized_vol"] = realized_vol
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalGapParams) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap_z = indicators["gap_z"].to_numpy(dtype=float)
        seasonal = indicators["seasonal_gap"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        rvol = indicators["realized_vol"].to_numpy(dtype=float)

        allow_short = bool(getattr(ctx, "allow_short", False))

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        pos = 0          # current position state (hysteresis: strong entry, trailing-stop exit)
        entry_idx = -1
        hwm = 0.0        # in-trade high-water close (long), ratchets up only
        lwm = 0.0        # in-trade low-water close (short), ratchets down only

        for i in range(n):
            gz = gap_z[i]
            sg = seasonal[i]
            a = atr[i]
            rv = rvol[i]

            entry_valid = not (np.isnan(gz) or np.isnan(sg) or np.isnan(a) or a <= 0.0)

            if (not np.isnan(rv)) and rv > 0.0:
                sz = params.target_vol / rv
            else:
                sz = 1.0
            sz = float(min(max(sz, params.size_floor), params.size_cap))

            if pos == 0:
                if entry_valid:
                    # Two-primitive AND: live gap z-score AND weekday gap climatology must agree.
                    long_ok = (gz >= params.gap_z_enter) and (sg >= params.seasonal_enter)
                    short_ok = (allow_short and gz <= -params.gap_z_enter
                                and sg <= -params.seasonal_enter)
                    if long_ok:
                        pos = 1
                        entry_idx = i
                        hwm = close[i]
                        signal[i] = 1
                        size[i] = sz
                    elif short_ok:
                        pos = -1
                        entry_idx = i
                        lwm = close[i]
                        signal[i] = -1
                        size[i] = sz
            elif pos == 1:
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.k_atr * a
                held = i - entry_idx
                if (close[i] <= stop) or (held >= params.max_hold):
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = sz
            else:  # pos == -1
                if close[i] < lwm:
                    lwm = close[i]
                stop = lwm + params.k_atr * a
                held = i - entry_idx
                if (close[i] >= stop) or (held >= params.max_hold):
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = -1
                    size[i] = sz

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0).clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
