from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownAreaParams:
    trend_window: int = 60
    area_window: int = 20
    atr_window: int = 14
    depth_thresh: float = 0.03
    drawup_thresh: float = 0.03
    purity_thresh: float = 0.25
    atr_mult: float = 3.0
    allow_short_flag: bool = True


class GeneratedStrategy(BaseStrategy[DrawdownAreaParams]):
    strategy_id = "gen_a1_1778907265"

    @classmethod
    def params_type(cls):
        return DrawdownAreaParams

    @staticmethod
    def warmup_bars(params: DrawdownAreaParams) -> int:
        return int(params.trend_window + params.area_window + params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownAreaParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_window), min_periods=1).mean()

        tw = int(params.trend_window)
        roll_max = close.rolling(tw, min_periods=1).max()
        roll_min = close.rolling(tw, min_periods=1).min()

        # instantaneous drawdown depth (<=0) and drawup from trough (>=0)
        dd = close / roll_max.replace(0.0, np.nan) - 1.0
        du = close / roll_min.replace(0.0, np.nan) - 1.0
        dd = dd.fillna(0.0)
        du = du.fillna(0.0)

        # integrated reservoirs: cumulative underwater / overwater area
        aw = int(params.area_window)
        uw_area = (-dd).clip(lower=0.0).rolling(aw, min_periods=1).sum()
        ow_area = (du).clip(lower=0.0).rolling(aw, min_periods=1).sum()

        total = (uw_area + ow_area).replace(0.0, np.nan)
        purity = ((ow_area - uw_area) / total).fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr.fillna(0.0)
        out["dd"] = dd
        out["du"] = du
        out["purity"] = purity
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownAreaParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        du = indicators["du"].to_numpy(dtype=float)
        purity = indicators["purity"].to_numpy(dtype=float)

        n = len(close)

        depth_t = float(params.depth_thresh)
        drawup_t = float(params.drawup_thresh)
        pur_t = float(params.purity_thresh)
        k = float(params.atr_mult)
        allow_short = bool(params.allow_short_flag)

        # two-primitive AND: instantaneous depth gate AND integrated trend-purity
        raw_long = (dd > -depth_t) & (purity > pur_t)
        raw_short = (du < drawup_t) & (purity < -pur_t)

        signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        stop_dist = 0.0

        for i in range(n):
            a = atr[i]
            if not np.isfinite(a) or a <= 0.0:
                a = 0.0
            cl = close[i]

            if position == 0:
                if raw_long[i]:
                    position = 1
                    entry_price = cl
                    stop_dist = k * a
                elif raw_short[i] and allow_short:
                    position = -1
                    entry_price = cl
                    stop_dist = k * a
            elif position == 1:
                stopped = (stop_dist > 0.0) and (cl < entry_price - stop_dist)
                if stopped or not raw_long[i]:
                    position = 0
                    if raw_short[i] and allow_short:
                        position = -1
                        entry_price = cl
                        stop_dist = k * a
            elif position == -1:
                stopped = (stop_dist > 0.0) and (cl > entry_price + stop_dist)
                if stopped or not raw_short[i]:
                    position = 0
                    if raw_long[i]:
                        position = 1
                        entry_price = cl
                        stop_dist = k * a

            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
