from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideTableParams:
    atr_window: int = 14
    norm_window: int = 63
    z_window: int = 126
    z_threshold: float = 1.0
    holding_bars: int = 18
    size_floor: float = 0.5
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[TideTableParams]):
    strategy_id = 'gen_1778829071'

    @classmethod
    def params_type(cls):
        return TideTableParams

    @staticmethod
    def warmup_bars(params: TideTableParams) -> int:
        return int(max(params.atr_window + 1, params.norm_window, params.z_window) + 504)

    @staticmethod
    def indicators(data: pd.DataFrame, params: TideTableParams) -> pd.DataFrame:
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        atr_baseline = atr.rolling(params.norm_window, min_periods=params.norm_window).mean()
        norm_atr = atr / atr_baseline.replace(0, np.nan)

        dom_values = np.asarray(data.index.day, dtype=int)
        dom = pd.Series(dom_values, index=data.index, name='dom')

        seasonal_frame = pd.DataFrame({'v': norm_atr, 'dom': dom}, index=data.index)
        # Per day-of-month expanding mean of normalized ATR, shifted within group to avoid lookahead.
        seasonal = seasonal_frame.groupby('dom')['v'].transform(
            lambda s: s.shift(1).expanding().mean()
        )

        residual = norm_atr - seasonal
        res_std = residual.rolling(params.z_window, min_periods=params.z_window).std()
        residual_z = residual / res_std.replace(0, np.nan)

        return pd.DataFrame({
            'atr': atr,
            'norm_atr': norm_atr,
            'seasonal': seasonal,
            'residual': residual,
            'residual_z': residual_z,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideTableParams,
    ) -> SignalFrame:
        rz = indicators['residual_z'].fillna(0.0)
        threshold = float(params.z_threshold)
        hb = int(params.holding_bars)
        if hb < 1:
            hb = 1

        long_trigger = (rz < -threshold).astype(int)
        short_trigger = (rz > threshold).astype(int)

        n = len(rz)
        idx_arr = np.arange(n, dtype=float)

        last_long = pd.Series(
            np.where(long_trigger.to_numpy() == 1, idx_arr, -np.inf),
            index=rz.index,
        ).cummax().to_numpy()
        last_short = pd.Series(
            np.where(short_trigger.to_numpy() == 1, idx_arr, -np.inf),
            index=rz.index,
        ).cummax().to_numpy()

        bars_since_long = idx_arr - last_long
        bars_since_short = idx_arr - last_short

        in_long = np.isfinite(last_long) & (bars_since_long < hb)
        in_short = np.isfinite(last_short) & (bars_since_short < hb)

        both = in_long & in_short
        raw_signal = np.where(
            both,
            np.where(bars_since_long <= bars_since_short, 1, -1),
            np.where(in_long, 1, np.where(in_short, -1, 0)),
        ).astype(int)

        raw_signal_s = pd.Series(raw_signal, index=data.index)

        trigger_mask = (long_trigger == 1) | (short_trigger == 1)
        size_raw = rz.abs().where(trigger_mask).ffill()
        size_raw = size_raw.clip(lower=params.size_floor, upper=params.size_cap).fillna(1.0)
        size_aligned = size_raw.where(raw_signal_s != 0, 1.0)

        signal_out = raw_signal_s.shift(1).fillna(0).astype(int)
        size_out = size_aligned.shift(1).fillna(1.0).clip(lower=1e-6).astype(float)

        df = pd.DataFrame(
            {'signal': signal_out, 'size': size_out},
            index=data.index,
        )

        return SignalFrame(data=df, signal_column='signal', size_column='size')
