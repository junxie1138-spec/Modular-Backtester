from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalElasticVolParams:
    atr_period: int = 14
    vol_period: int = 20
    elastic_factor: float = 0.85
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 18
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[SeasonalElasticVolParams]):
    strategy_id = "gen_a1_1778893446"

    @classmethod
    def params_type(cls):
        return SeasonalElasticVolParams

    def warmup_bars(self, params):
        return int(max(params.atr_period, params.vol_period)) + 2

    def indicators(self, data, params):
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        log_ret = np.log(close / close.shift(1))
        out["log_ret"] = log_ret

        # realized volatility: rolling std of log returns
        rv = log_ret.rolling(p.vol_period, min_periods=p.vol_period).std()
        out["rv"] = rv

        # ATR (simple mean of true range)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()
        out["atr"] = atr

        # month-of-year climatology, trailing only (shift(1) inside each group)
        moy = pd.Series(data.index.month, index=data.index)

        clim_vol = rv.groupby(moy).transform(
            lambda s: s.shift(1).expanding(min_periods=20).mean()
        )
        out["clim_vol"] = clim_vol

        clim_ret = log_ret.groupby(moy).transform(
            lambda s: s.shift(1).expanding(min_periods=20).mean()
        )
        out["clim_ret"] = clim_ret

        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        close = data["close"].to_numpy(dtype=float)
        rv = indicators["rv"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        clim_vol = indicators["clim_vol"].to_numpy(dtype=float)
        clim_ret = indicators["clim_ret"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, float(p.base_size), dtype=float)

        # primitive A (volatility): elastic regime - current vol below the
        # calendar-month climatological norm
        threshold = clim_vol * p.elastic_factor
        elastic = (rv > 0.0) & (clim_vol > 0.0) & (rv < threshold)
        # primitive B (seasonality): month seasonally bullish
        favorable = clim_ret > 0.0
        # two-primitive AND - both must agree
        entry = elastic & favorable

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                signal[i] = 1
                bars_held += 1
                exit_now = False
                # fixed volatility-stop: stop fixed at entry, not trailing
                if not np.isnan(close[i]) and close[i] < stop_level:
                    exit_now = True
                elif bars_held >= p.max_hold_bars:
                    exit_now = True
                if exit_now:
                    in_pos = False
                    signal[i] = 0
                    bars_held = 0
            else:
                valid = (
                    not np.isnan(close[i])
                    and not np.isnan(atr[i])
                    and not np.isnan(rv[i])
                    and not np.isnan(clim_vol[i])
                    and not np.isnan(clim_ret[i])
                )
                if valid and bool(entry[i]) and atr[i] > 0.0:
                    in_pos = True
                    stop_level = close[i] - p.atr_stop_mult * atr[i]
                    bars_held = 0
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
