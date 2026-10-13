from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeekdayShockParams:
    season_window: int = 126
    min_samples: int = 12
    shock_mult: float = 1.3
    rejection: float = 0.7
    atr_window: int = 14
    stop_atr: float = 1.0
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[WeekdayShockParams]):
    strategy_id = 'gen_local-w2_1791287822'

    @classmethod
    def params_type(cls):
        return WeekdayShockParams

    def warmup_bars(self, params: WeekdayShockParams) -> int:
        return int(max(params.season_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WeekdayShockParams) -> pd.DataFrame:
        if not (params.season_window >= params.min_samples >= 2 and params.atr_window >= 1 and params.shock_mult > 0 and params.stop_atr > 0 and 0.5 < params.rejection <= 1.0 and params.hold_days in (1, 2)):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['weekday'] = data.index.dayofweek
        for day in range(5):
            out[f'weekday_std_{day}'] = out['return'].where(out['weekday'].eq(day)).rolling(params.season_window, min_periods=params.min_samples).std(ddof=1).shift(1)
        out['weekday_std'] = pd.concat([
            out[f'weekday_std_{day}'].where(out['weekday'].eq(day))
            for day in range(5)
        ], axis=1).sum(axis=1, min_count=1)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_location'] = ((data['close'] - data['low']) / out['range']).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WeekdayShockParams) -> SignalFrame:
        valid = indicators[['weekday_std', 'atr', 'return', 'close_location']].apply(np.isfinite).all(axis=1)
        valid &= (indicators['weekday_std'] > 0) & (indicators['atr'] > 0)
        long_entry = valid & (indicators['return'] < -params.shock_mult * indicators['weekday_std']) & (indicators['close_location'] >= params.rejection)
        short_entry = valid & (indicators['return'] > params.shock_mult * indicators['weekday_std']) & (indicators['close_location'] <= 1.0 - params.rejection)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        target = np.zeros(len(data), dtype=int)
        position, held = 0, 0
        entry, frozen_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                # This is the fill bar following the prior close's entry decision.
                if held == 0:
                    entry = opens[i]
                held += 1
                stopped = (position == 1 and closes[i] < entry - params.stop_atr * frozen_atr) or (position == -1 and closes[i] > entry + params.stop_atr * frozen_atr)
                if not np.isfinite(entry) or not np.isfinite(closes[i]) or stopped or held >= params.hold_days:
                    position, held = 0, 0
                target[i] = position
                continue
            if candidates[i] and np.isfinite(closes[i]):
                position = int(candidates[i])
                frozen_atr = atrs[i]
                held = 0
            target[i] = position
        df = pd.DataFrame({'signal': target, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.bdate_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': [100, 98, 100, 100, 103, 100, 100, 100, 100, 100], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'return': [-2, 0, 0, 2, 0, 0, -2, 0, 0, 0], 'weekday_std': 1.0, 'atr': 1.0, 'close_location': [0.9, 0.5, 0.5, 0.1, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5]}, index=index)
    strategy = GeneratedStrategy()
    params = WeekdayShockParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1, 0]
    for length in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:length], indicators.iloc[:length], ctx, params).data
        assert prefix.equals(result.iloc[:length])
