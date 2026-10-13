from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreakoutParams:
    volume_window: int = 20
    area_window: int = 12
    breakout_window: int = 20
    atr_window: int = 14
    area_threshold: float = 0.08
    stop_atr: float = 2.8
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.volume_window, self.area_window, self.breakout_window, self.atr_window, self.holding_bars) < 2:
            raise ValueError('Lookbacks and holding_bars must be at least two.')
        if not (np.isfinite(self.area_threshold) and 0 < self.area_threshold < 0.5):
            raise ValueError('area_threshold must lie between zero and 0.5.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[SignatureBreakoutParams]):
    strategy_id = 'gen_local_1790897100'

    @classmethod
    def params_type(cls):
        return SignatureBreakoutParams

    def warmup_bars(self, params: SignatureBreakoutParams) -> int:
        return int(max(params.volume_window + params.area_window + 1,
                       params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureBreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volume_baseline'] = data['volume'].shift(1).rolling(params.volume_window).mean()
        out['volume_impulse'] = (data['volume'] / out['volume_baseline'].where(out['volume_baseline'] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out['volume_path'] = out['volume_impulse'].fillna(0).cumsum()
        out['return_path'] = out['return'].fillna(0).cumsum()
        out['cross_increment'] = (out['volume_path'].shift(1).fillna(0) * out['return']
                                  - out['return_path'].shift(1).fillna(0) * out['volume_impulse'])
        length = params.area_window
        out['volume_displacement'] = out['volume_impulse'].rolling(length).sum()
        out['return_displacement'] = out['return'].rolling(length).sum()
        out['signature_area'] = 0.5 * (
            out['cross_increment'].rolling(length).sum()
            - out['volume_path'].shift(length) * out['return_displacement']
            + out['return_path'].shift(length) * out['volume_displacement']
        )
        out['area_scale'] = (out['volume_impulse'].abs().rolling(length).sum()
                             * out['return'].abs().rolling(length).sum())
        out['normalized_area'] = (out['signature_area'] / out['area_scale'].where(out['area_scale'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['prior_high'] = data['high'].shift(1).rolling(params.breakout_window).max()
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['area_agrees'] = out['normalized_area'].gt(params.area_threshold).fillna(False)
        out['breakout_agrees'] = data['close'].gt(out['prior_high']).fillna(False)
        out['entry'] = (out['area_agrees'] & out['breakout_agrees']
                        & out['atr'].gt(0) & np.isfinite(out['atr']))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureBreakoutParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        eligible = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The pending entry fills at this bar's open; ATR was known at the previous close.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                stopped = np.isfinite(close[i]) and np.isfinite(fixed_stop) and close[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired or not np.isfinite(fixed_stop):
                    active = False
                else:
                    decisions[i] = 1
            elif eligible[i] and np.isfinite(close[i]):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                fixed_stop = np.nan
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Check next-open entry pricing, a frozen stop, and exactly one bar of signal delay.
    index = pd.date_range('2020-01-01', periods=6)
    sample = pd.DataFrame({'open': [90, 100, 100, 110, 95, 93],
                           'high': [92, 102, 112, 111, 96, 94],
                           'low': [89, 99, 99, 94, 92, 92],
                           'close': [91, 101, 110, 95, 93, 93],
                           'volume': [100.0] * 6}, index=index)
    derived = pd.DataFrame({'atr': [2.0] * 6,
                            'entry': [True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, sample, derived, None,
                                                SignatureBreakoutParams(stop_atr=3.0))
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result.data['size'].gt(0).all()
