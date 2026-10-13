from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_bars: int = 63
    history_bars: int = 504
    hold_bars: int = 8
    min_samples: int = 16
    min_drawdown: float = 0.012
    forecast_threshold: float = 0.003
    full_size_forecast: float = 0.02
    slope_ridge: float = 0.0001
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790997661'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(2, params.peak_bars) + max(2, params.history_bars) + max(1, params.hold_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        peak_bars = max(2, int(params.peak_bars))
        history = max(2, int(params.history_bars))
        horizon = max(1, int(params.hold_bars))
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        out['peak'] = close.rolling(peak_bars, min_periods=peak_bars).max()
        out['drawdown'] = (1.0 - close / out['peak']).clip(lower=0.0)
        out['month'] = pd.Series(data.index.month, index=data.index, dtype=float)
        out['sample_drawdown'] = out['drawdown'].shift(horizon)
        out['sample_month'] = out['month'].shift(horizon)
        # Each historical outcome ends on the current or an earlier close.
        out['matured_return'] = close / close.shift(horizon) - 1.0
        x = out['sample_drawdown']
        y = out['matured_return'].replace([np.inf, -np.inf], np.nan)
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        for month in range(1, 13):
            valid = out['sample_month'].eq(month) & x.notna() & y.notna()
            count = valid.astype(float).rolling(history, min_periods=1).sum()
            denominator = count.replace(0.0, np.nan)
            mean_x = x.where(valid, 0.0).rolling(history, min_periods=1).sum() / denominator
            mean_y = y.where(valid, 0.0).rolling(history, min_periods=1).sum() / denominator
            mean_xx = (x * x).where(valid, 0.0).rolling(history, min_periods=1).sum() / denominator
            mean_xy = (x * y).where(valid, 0.0).rolling(history, min_periods=1).sum() / denominator
            variance = (mean_xx - mean_x * mean_x).clip(lower=0.0)
            covariance = mean_xy - mean_x * mean_y
            beta = covariance / (variance + max(1e-12, float(params.slope_ridge)))
            prediction = mean_y + beta * (out['drawdown'] - mean_x)
            ready = count.ge(max(2, int(params.min_samples)))
            prediction = prediction.where(ready).replace([np.inf, -np.inf], np.nan)
            prefix = 'month_' + str(month) + '_'
            out[prefix + 'count'] = count
            out[prefix + 'mean_drawdown'] = mean_x
            out[prefix + 'mean_return'] = mean_y
            out[prefix + 'mean_squared_drawdown'] = mean_xx
            out[prefix + 'mean_cross_product'] = mean_xy
            out[prefix + 'variance'] = variance
            out[prefix + 'covariance'] = covariance
            out[prefix + 'beta'] = beta
            out[prefix + 'forecast'] = prediction
            forecast = forecast.where(out['month'].ne(month), prediction)
        out['forecast'] = forecast
        eligible = out['drawdown'].ge(max(0.0, float(params.min_drawdown)))
        eligible &= forecast.abs().ge(max(1e-12, float(params.forecast_threshold)))
        eligible &= close.notna() & forecast.notna()
        out['candidate'] = np.sign(forecast.fillna(0.0)).where(eligible, 0).astype(int)
        lower = min(1.0, max(0.01, float(params.min_size)))
        upper = min(1.0, max(lower, float(params.max_size)))
        out['conviction_size'] = (forecast.abs() / max(1e-12, float(params.full_size_forecast))).clip(lower=lower, upper=upper).fillna(lower)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        lower = min(1.0, max(0.01, float(params.min_size)))
        sizes = np.full(count, lower, dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        convictions = indicators['conviction_size'].fillna(lower).to_numpy(dtype=float)
        horizon = max(1, int(params.hold_bars))
        position = 0
        entry_size = lower
        exit_bar = -1
        for i in range(count):
            if position != 0:
                if i >= exit_bar:
                    # Reserve this bar for the fixed exit, even if an entry qualifies.
                    position = 0
                    entry_size = lower
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_size = float(convictions[i])
                exit_bar = i + horizon
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        # Shift both direction and frozen entry sizing to the execution bar.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(lower).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
