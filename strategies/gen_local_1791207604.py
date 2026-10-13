from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    baseline_window: int = 40
    recent_window: int = 5
    reference_window: int = 15
    expansion_multiple: float = 1.25
    prevalence_jump: float = 0.30
    pressure_threshold: float = 0.20
    atr_window: int = 14
    trailing_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791207604'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.baseline_window + params.recent_window + params.reference_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['range_baseline'] = out['range'].shift(1).rolling(p.baseline_window, min_periods=p.baseline_window).median()
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['midpoint_change'] = out['midpoint'].diff()
        valid = out['range_baseline'].gt(0) & out['midpoint_change'].notna() & out['range'].notna()
        out['expanding'] = out['range'].gt(p.expansion_multiple * out['range_baseline']).astype(float).where(valid)
        out['up_expansion'] = (out['expanding'] * out['midpoint_change'].gt(0).astype(float)).where(valid)
        out['down_expansion'] = (out['expanding'] * out['midpoint_change'].lt(0).astype(float)).where(valid)
        for direction in ('up', 'down'):
            events = out[direction + '_expansion']
            out[direction + '_recent'] = events.rolling(p.recent_window, min_periods=p.recent_window).mean()
            out[direction + '_reference'] = events.shift(p.recent_window).rolling(p.reference_window, min_periods=p.reference_window).mean()
            out[direction + '_jump'] = out[direction + '_recent'] - out[direction + '_reference']
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['range'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out['pressure'] = out['close_location'].rolling(p.recent_window, min_periods=p.recent_window).mean()
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([out['range'], (data['high'] - out['previous_close']).abs(), (data['low'] - out['previous_close']).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        ready = out[['up_jump', 'down_jump', 'pressure', 'atr']].notna().all(axis=1) & out['atr'].gt(0)
        bullish = out['up_jump'].ge(p.prevalence_jump) & out['up_jump'].gt(out['down_jump']) & out['pressure'].ge(p.pressure_threshold)
        bearish = out['down_jump'].ge(p.prevalence_jump) & out['down_jump'].gt(out['up_jump']) & out['pressure'].le(-p.pressure_threshold)
        out['entry'] = np.select([ready & bullish, ready & bearish], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = low_water = 0.0
        lower_stop = upper_stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0.0 or atr <= 0.0:
                position = 0
                continue
            if position:
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                lower_stop = max(lower_stop, high_water - params.trailing_atr * atr)
                # The required rolling-high stop applies to both directions.
                # Shorts also have a mirrored stop against adverse rebounds.
                if position == -1:
                    upper_stop = min(upper_stop, low_water + params.trailing_atr * atr)
                stopped = close <= lower_stop or (position == -1 and close >= upper_stop)
                expired = i - entry_bar >= params.holding_bars
                if stopped or expired:
                    position = 0
                    continue
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                entry_bar = i
                high_water = low_water = close
                lower_stop = close - params.trailing_atr * atr
                upper_stop = close + params.trailing_atr * atr
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'close': [100.0, 105.0, 104.0, 103.0, 102.0, 100.0, 99.0, 101.0, 104.0, 104.0]}, index=index)
    indicators = pd.DataFrame({'atr': [2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 'entry': [1, 0, 0, 0, 0, 1, -1, 0, 0, 0]}, index=index)
    p = StrategyParams(trailing_atr=2.0)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, -1, -1, 0]
    p.holding_bars = 2
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].iloc[3] == 0
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
