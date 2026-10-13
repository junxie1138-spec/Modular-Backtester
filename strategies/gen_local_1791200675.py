from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnergyParams:
    energy_window: int = 6
    history_window: int = 126
    atr_window: int = 14
    entry_percentile: float = 0.85
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.energy_window, self.history_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must lie between 0.5 and 1.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or self.hold_bars < 1:
            raise ValueError('Stop distance and holding period must be positive.')


class GeneratedStrategy(BaseStrategy[EnergyParams]):
    strategy_id = 'gen_local_1791200675'

    @classmethod
    def params_type(cls):
        return EnergyParams

    def warmup_bars(self, params: EnergyParams) -> int:
        return int(max(params.energy_window + params.history_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['return'] = close.div(out['previous_close'].where(out['previous_close'] > 0)).sub(1)
        out['return'] = out['return'].replace([np.inf, -np.inf], np.nan)
        r = out['return']
        window = params.energy_window
        out['up_energy'] = r.clip(lower=0).pow(2).rolling(window, min_periods=window).mean()
        out['down_energy'] = r.clip(upper=0).pow(2).rolling(window, min_periods=window).mean()
        out['total_energy'] = out['up_energy'] + out['down_energy']
        out['dominance'] = (out['up_energy'] - out['down_energy']).div(
            out['total_energy'].where(out['total_energy'] > 0))
        out['volatility'] = r.rolling(window, min_periods=window).std(ddof=0)
        out['momentum'] = r.rolling(window, min_periods=window).sum()
        history = params.history_window
        dominance_history = out['dominance'].rolling(history, min_periods=history)
        out['upper_percentile'] = dominance_history.quantile(params.entry_percentile).shift(1)
        out['lower_percentile'] = dominance_history.quantile(1.0 - params.entry_percentile).shift(1)
        out['volatility_median'] = out['volatility'].rolling(
            history, min_periods=history).quantile(0.5).shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        required = ['dominance', 'volatility', 'momentum', 'upper_percentile',
                    'lower_percentile', 'volatility_median', 'atr']
        out['ready'] = np.isfinite(out[required]).all(axis=1) & out['atr'].gt(0)
        out['expansion'] = out['volatility'].gt(out['volatility_median'])
        out['long_entry'] = (
            out['ready'] & out['expansion'] & out['momentum'].gt(0)
            & out['dominance'].gt(0)
            & out['dominance'].gt(out['upper_percentile'])
            & out['dominance'].shift(1).le(out['upper_percentile'].shift(1))
        ).fillna(False)
        out['short_entry'] = (
            out['ready'] & out['expansion'] & out['momentum'].lt(0)
            & out['dominance'].lt(0)
            & out['dominance'].lt(out['lower_percentile'])
            & out['dominance'].shift(1).ge(out['lower_percentile'].shift(1))
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EnergyParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # The preceding close requested entry; this bar's open is the fill reference.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    fixed_stop = opens[i] - side * params.stop_atr * entry_atr
                invalid_close = not np.isfinite(close[i]) or close[i] <= 0
                stopped = (close[i] < fixed_stop if side == 1 else close[i] > fixed_stop)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid_close or stopped or expired:
                    side = 0
                    continue
            elif np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i] or shorts[i]:
                    side = 1 if longs[i] else -1
                    entry_bar = i + 1
                    entry_atr = atr[i]
            decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = EnergyParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    marks = pd.DataFrame({'atr': 2.0, 'long_entry': False,
                          'short_entry': False}, index=index)
    marks.iloc[0, marks.columns.get_loc('long_entry')] = True
    result = GeneratedStrategy.generate_signals(None, data, marks, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 3
    assert result['size'].gt(0).all()
    # Both stops use the next open and freeze the ATR from the decision bar.
    data.loc[index[0], 'close'] = 80.0
    data.loc[index[1], 'close'] = 95.0
    data.loc[index[3], 'close'] = 105.0
    marks.loc[index[1]:, 'atr'] = 100.0
    marks.loc[index[2], 'atr'] = 2.0
    marks.loc[index[2], 'short_entry'] = True
    result = GeneratedStrategy.generate_signals(None, data, marks, None, params).data
    assert result['signal'].iloc[:5].tolist() == [0, 1, 0, -1, 0]
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived.index.equals(data.index)
    assert not derived['long_entry'].any() and not derived['short_entry'].any()
    assert GeneratedStrategy.warmup_bars(None, params) >= 134


if __name__ == '__main__':
    _self_check()
