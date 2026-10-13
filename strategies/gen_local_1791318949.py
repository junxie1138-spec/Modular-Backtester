from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    trend_period: int = 5
    slope_period: int = 5
    history_bars: int = 126
    min_samples: int = 12
    entry_quantile: float = 0.8
    atr_period: int = 14
    trail_atr: float = 1.25
    max_hold: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791318949'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.trend_period + params.slope_period + params.history_bars + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (params.trend_period >= 2 and params.slope_period >= 2 and params.history_bars >= params.min_samples >= 2 and params.atr_period >= 1):
            raise ValueError('Invalid lookbacks')
        if not (0 < params.entry_quantile < 1 and np.isfinite(params.trail_atr) and params.trail_atr > 0 and params.max_hold in (1, 2) and np.isfinite(params.size) and params.size > 0):
            raise ValueError('Invalid threshold, stop, holding period, or size')
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        out['log_close'] = np.log(close.where(close > 0))
        out['trend'] = out['log_close'].rolling(params.trend_period).mean()
        out['slope'] = out['trend'].diff()
        out['predicted_slope'] = out['slope'].shift(1).rolling(params.slope_period).mean()
        out['innovation'] = out['slope'] - out['predicted_slope']
        out['log_return'] = out['log_close'].diff()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).replace([np.inf, -np.inf], np.nan).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period).mean()
        out['phase'] = pd.Series(np.minimum((data.index.day - 1) // 10, 2), index=data.index)
        out['previous_phase'] = out['phase'].shift(1)
        out['innovation_threshold'] = np.nan
        out['seasonal_drift'] = np.nan
        for phase in range(3):
            # Exclude today's innovation from its reference distribution.
            history = out['innovation'].where(out['phase'].eq(phase)).shift(1)
            out[f'threshold_{phase}'] = history.rolling(params.history_bars, min_periods=params.min_samples).quantile(params.entry_quantile)
            # Today's realized return labels yesterday's calendar phase.
            outcomes = out['log_return'].where(out['previous_phase'].eq(phase))
            out[f'drift_{phase}'] = outcomes.rolling(params.history_bars, min_periods=params.min_samples).mean()
            mask = out['phase'].eq(phase)
            out.loc[mask, 'innovation_threshold'] = out.loc[mask, f'threshold_{phase}']
            out.loc[mask, 'seasonal_drift'] = out.loc[mask, f'drift_{phase}']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        entry = (
            indicators['innovation'].gt(indicators['innovation_threshold'])
            & indicators['innovation'].gt(0)
            & indicators['seasonal_drift'].gt(0)
            & indicators['atr'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                age += 1
                if valid:
                    # The first close here belongs to the actual entry session.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                if not valid or closes[i] <= stop or age >= params.max_hold:
                    active = False
                    age = 0
                    high_water = -np.inf
                    stop = -np.inf
                else:
                    raw[i] = 1
            elif valid and i >= self.warmup_bars(params) - 1 and entry[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GeneratedParams(trend_period=2, slope_period=2, history_bars=2, min_samples=2, atr_period=1)
    strategy = GeneratedStrategy()
    start = strategy.warmup_bars(params) - 1
    index = pd.date_range('2020-01-01', periods=start + 5, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'innovation': 0.0, 'innovation_threshold': 0.5, 'seasonal_drift': 0.001, 'atr': 1.0}, index=index)
    indicators.iloc[start, indicators.columns.get_loc('innovation')] = 1.0
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].iloc[start:].tolist() == [0, 1, 1, 0, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
