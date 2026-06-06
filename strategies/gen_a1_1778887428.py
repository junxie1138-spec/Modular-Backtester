from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticBreakoutParams:
    ema_period: int = 20
    atr_period: int = 14
    band_mult: float = 1.5
    thrust_window: int = 5
    thrust_threshold: float = 1.0
    breakeven_pct: float = 0.03
    trail_atr_mult: float = 2.5
    base_size: float = 0.5
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ElasticBreakoutParams]):
    strategy_id = "gen_a1_1778887428"

    @classmethod
    def params_type(cls):
        return ElasticBreakoutParams

    def warmup_bars(self, params: ElasticBreakoutParams) -> int:
        return int(max(params.ema_period, params.atr_period, params.thrust_window)) + 2

    def indicators(self, data: pd.DataFrame, params: ElasticBreakoutParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        atr_safe = atr.replace(0.0, np.nan)

        ema = close.ewm(span=params.ema_period, adjust=False).mean()
        upper_band = ema + params.band_mult * atr

        daily_move = close.diff()
        norm_move = daily_move / atr_safe
        pos_thrust = norm_move.clip(lower=0.0)
        thrust = pos_thrust.rolling(
            params.thrust_window, min_periods=params.thrust_window
        ).sum()

        out = pd.DataFrame(index=data.index)
        out["ema"] = ema
        out["atr"] = atr
        out["upper_band"] = upper_band
        out["thrust"] = thrust
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticBreakoutParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        upper = indicators["upper_band"].to_numpy(dtype=float)
        thrust = indicators["thrust"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size, dtype=float)

        span = max(params.max_size - params.base_size, 0.0)
        thr = max(params.thrust_threshold, 1e-9)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        highest = 0.0
        be_hit = False
        cur_size = params.base_size

        for i in range(n):
            ci = close[i]
            ai = atr[i]
            ui = upper[i]
            ti = thrust[i]
            valid = (
                not (np.isnan(ci) or np.isnan(ai) or np.isnan(ui) or np.isnan(ti))
                and ai > 0.0
            )

            if not in_pos:
                if valid and ci > ui and ti > params.thrust_threshold:
                    in_pos = True
                    entry_price = ci
                    highest = ci
                    be_hit = False
                    stop = ci - params.trail_atr_mult * ai
                    conv = (ti - params.thrust_threshold) / thr
                    conv = min(max(conv, 0.0), 1.0)
                    cur_size = params.base_size + conv * span
                    signal[i] = 1
                    size[i] = cur_size
            else:
                if not np.isnan(ci) and ci > highest:
                    highest = ci
                if not np.isnan(ci) and not be_hit and ci >= entry_price * (
                    1.0 + params.breakeven_pct
                ):
                    be_hit = True
                    if entry_price > stop:
                        stop = entry_price
                if be_hit and valid:
                    trail = highest - params.trail_atr_mult * ai
                    if trail > stop:
                        stop = trail
                if not np.isnan(ci) and ci <= stop:
                    in_pos = False
                    be_hit = False
                    signal[i] = 0
                    size[i] = cur_size
                else:
                    signal[i] = 1
                    size[i] = cur_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
