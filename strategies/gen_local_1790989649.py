from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    reclaim_window: int = 20
    trend_span: int = 12
    slope_span: int = 6
    innovation_span: int = 3
    atr_window: int = 14
    min_drawdown: float = 0.025
    innovation_threshold: float = 0.04
    reclaim_quantile: float = 0.70
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1790989649'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.peak_window + params.reclaim_window - 1,
            params.trend_span + params.slope_span + params.innovation_span + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        scale = atr.where(atr > 0.0)
        level = close.ewm(
            span=params.trend_span, adjust=False,
            min_periods=params.trend_span,
        ).mean()
        observed_slope = level.diff()
        latent_slope = observed_slope.ewm(
            span=params.slope_span, adjust=False,
            min_periods=params.slope_span,
        ).mean()
        slope_residual = observed_slope - latent_slope.shift(1)
        slope_innovation = slope_residual.ewm(
            span=params.innovation_span, adjust=False,
            min_periods=params.innovation_span,
        ).mean() / scale
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = 1.0 - close / peak.where(peak > 0.0)
        episode_depth = drawdown.rolling(
            params.reclaim_window, min_periods=params.reclaim_window,
        ).max()
        underwater_close = close.where(drawdown >= params.min_drawdown * 0.25)
        required = max(2, params.reclaim_window // 2)
        underwater_low = underwater_close.rolling(
            params.reclaim_window, min_periods=required,
        ).quantile(1.0 - params.reclaim_quantile)
        underwater_high = underwater_close.rolling(
            params.reclaim_window, min_periods=required,
        ).quantile(params.reclaim_quantile)
        result = pd.DataFrame({
            'true_range': true_range,
            'atr': atr,
            'latent_level': level,
            'observed_slope': observed_slope,
            'latent_slope': latent_slope,
            'slope_residual': slope_residual,
            'slope_innovation': slope_innovation,
            'rolling_peak': peak,
            'drawdown': drawdown,
            'episode_depth': episode_depth,
            'underwater_close': underwater_close,
            'underwater_low': underwater_low,
            'underwater_high': underwater_high,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        innovation = indicators['slope_innovation'].to_numpy(dtype=float)
        lower = indicators['underwater_low'].to_numpy(dtype=float)
        upper = indicators['underwater_high'].to_numpy(dtype=float)
        depth = indicators['episode_depth'].to_numpy(dtype=float)
        ready = (
            np.isfinite(close) & np.isfinite(atr) & (atr > 0.0)
            & np.isfinite(innovation) & np.isfinite(lower)
            & np.isfinite(upper) & np.isfinite(depth)
        )
        episode = depth >= params.min_drawdown
        long_entry = (
            ready & episode
            & (innovation >= params.innovation_threshold)
            & (close > upper)
        )
        short_entry = (
            ready & episode
            & (innovation <= -params.innovation_threshold)
            & (close < lower)
        )
        signals = np.zeros(len(data), dtype=np.int64)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if i < warmup - 1:
                continue
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                position = 0
                age = 0
                continue

            if position != 0:
                # The entry decision filled on this or an earlier bar.
                # Initialize water marks using closes after that fill.
                age += 1
                high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                low_water = close[i] if np.isnan(low_water) else min(low_water, close[i])
                high_stop = max(high_stop, high_water - params.trail_atr * atr[i])
                if position == -1:
                    short_stop = min(short_stop, low_water + params.trail_atr * atr[i])
                # Every trade has the required ratcheting closing-high exit.
                # Shorts also have a symmetric rebound protection exit.
                exit_now = (
                    close[i] <= high_stop
                    or (position == -1 and close[i] >= short_stop)
                    or age >= params.max_hold_bars
                )
                if exit_now:
                    position = 0
                    age = 0
                    continue
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
                high_water = np.nan
                low_water = np.nan
                high_stop = -np.inf
                short_stop = np.inf

            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
