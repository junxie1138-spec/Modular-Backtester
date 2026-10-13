from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    migration_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.8
    profit_target: float = 0.04
    max_hold_bars: int = 20

    def __post_init__(self):
        if self.migration_window < 2 or self.percentile_window < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one.')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be finite and positive.')
        if self.max_hold_bars < 1:
            raise ValueError('max_hold_bars must be positive.')


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = 'gen_local_1791302549'

    @classmethod
    def params_type(cls):
        return RangeMigrationParams

    def warmup_bars(self, params):
        return int(params.migration_window + params.percentile_window + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = high.ge(low) & low.gt(0.0)
        high = high.where(valid)
        low = low.where(valid)
        out['daily_range'] = high - low
        out['high_step'] = high.diff()
        out['low_step'] = low.diff()
        # Count only movement shared by both boundaries; expanding ranges alone earn no credit.
        out['up_translation'] = np.minimum(out['high_step'], out['low_step']).clip(lower=0.0)
        out['down_translation'] = (-np.maximum(out['high_step'], out['low_step'])).clip(lower=0.0)
        out['net_translation'] = out['up_translation'] - out['down_translation']
        window = params.migration_window
        out['translation_sum'] = out['net_translation'].rolling(window, min_periods=window).sum()
        out['range_sum'] = out['daily_range'].rolling(window, min_periods=window).sum()
        out['strength'] = out['translation_sum'].div(out['range_sum'].where(out['range_sum'].gt(0.0)))
        out['threshold'] = out['strength'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile)
        out['qualified'] = (
            out['strength'].notna()
            & out['threshold'].notna()
            & out['strength'].gt(0.0)
            & out['strength'].gt(out['threshold'])
        )
        out['entry'] = out['qualified'] & ~out['qualified'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1

        for i in range(count):
            if active and entry_bar == -1:
                # The preceding close requested entry; this bar supplies the executable open.
                entry_bar = i
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    active = False
                    continue

            if active:
                # ponytail: gross opening-price reference; use actual fills if the contract exposes them.
                profit_exit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_exit = i - entry_bar + 1 >= params.max_hold_bars
                if profit_exit or time_exit:
                    active = False
                    entry_price = np.nan
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_bar = -1
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [106.0] * 7,
        'low': [99.0] * 7,
        'close': [100.0, 100.0, 105.0, 100.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False, False, False]}, index=index)
    params = RangeMigrationParams(max_hold_bars=4)
    profit = strategy.generate_signals(data, indicators, None, params).data
    assert profit['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data['close'] = 100.0
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert timed['size'].gt(0.0).all()
    assert not strategy.indicators(data, params)['entry'].any()
    assert strategy.warmup_bars(params) == 139


if __name__ == '__main__':
    _self_check()
