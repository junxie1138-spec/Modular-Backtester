from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    ma_window: int = 20
    shape_window: int = 6
    history_bars: int = 252
    hold_bars: int = 4
    analog_radius: float = 0.8
    min_analogs: int = 8
    min_edge: float = 0.003
    rejection_window: int = 3
    rejection_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = 'gen_local-w2_1791302633'

    @classmethod
    def params_type(cls) -> type[ExcursionParams]:
        return ExcursionParams

    @staticmethod
    def warmup_bars(params: ExcursionParams) -> int:
        return int(max(params.ma_window + params.shape_window + params.history_bars + 1, params.rejection_window))

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        p = params
        if not (p.ma_window >= 2 and p.shape_window >= 2 and 3 <= p.hold_bars <= 5):
            raise ValueError('Invalid moving-average, shape, or holding window')
        first_lag = p.ma_window + p.shape_window + p.hold_bars + 1
        if not (p.history_bars >= first_lag and p.rejection_window >= 1):
            raise ValueError('Insufficient analog history or invalid rejection window')
        if not (np.isfinite(p.analog_radius) and p.analog_radius > 0 and 1 <= p.min_analogs <= p.history_bars - first_lag + 1):
            raise ValueError('Invalid analog radius or minimum count')
        if not (np.isfinite(p.min_edge) and p.min_edge >= 0 and np.isfinite(p.rejection_threshold) and 0 <= p.rejection_threshold <= 1):
            raise ValueError('Invalid entry thresholds')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['ma'] = close.rolling(p.ma_window).mean()
        out['price_std'] = close.rolling(p.ma_window).std(ddof=0).replace(0, np.nan)
        out['z'] = (close - out['ma']) / out['price_std']
        out['underwater_area'] = (-out['z']).clip(lower=0).rolling(p.shape_window).mean()
        out['trough_z'] = out['z'].rolling(p.shape_window).min()
        out['recovery'] = out['z'] - out['trough_z']
        features = out[['z', 'underwater_area', 'recovery']]
        weighted_return = pd.Series(0.0, index=data.index)
        weight_sum = pd.Series(0.0, index=data.index)
        count = pd.Series(0, index=data.index, dtype=int)

        # All analog entry and exit opens are observed before the current decision.
        # ponytail: linear scan over a bounded history; batch distances if histories grow.
        for lag in range(first_lag, p.history_bars + 1):
            distance = (features - features.shift(lag)).pow(2).sum(axis=1, min_count=3)
            outcome = opening.shift(lag - p.hold_bars - 1) / opening.shift(lag - 1) - 1.0
            valid = (distance <= p.analog_radius ** 2) & np.isfinite(distance) & np.isfinite(outcome)
            weight = (1.0 / (1.0 + distance)).where(valid, 0.0)
            weighted_return += weight * outcome.where(valid, 0.0)
            weight_sum += weight
            count += valid.astype(int)

        out['analog_count'] = count
        out['analog_weight'] = weight_sum
        out['analog_edge'] = weighted_return / weight_sum.replace(0, np.nan)
        out['bar_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['lower_wick'] = data[['open', 'close']].min(axis=1, skipna=False) - data['low']
        out['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1, skipna=False)
        out['wick_balance'] = (out['lower_wick'] - out['upper_wick']) / out['bar_range']
        out['rejection'] = out['wick_balance'].rolling(p.rejection_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExcursionParams) -> SignalFrame:
        p = params
        eligible = (
            (indicators['z'] < 0)
            & (indicators['recovery'] > 0)
            & (indicators['analog_count'] >= p.min_analogs)
            & (indicators['analog_edge'] > p.min_edge)
            & (indicators['rejection'] > p.rejection_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        eligible[:self.warmup_bars(p)] = False
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force one flat target; entry conditions cannot postpone exit.
                    exit_bar = -1
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + p.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    p = ExcursionParams()
    strategy = GeneratedStrategy()
    warmup = strategy.warmup_bars(p)
    index = pd.date_range('2000-01-01', periods=warmup + 15)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'z': -1.0, 'recovery': 0.5, 'analog_count': 20, 'analog_edge': 0.02, 'rejection': 0.5}, index=index)
    result = strategy.generate_signals(data, indicators, None, p).data
    expected = [0] * (warmup + 1) + [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    assert result['signal'].tolist() == expected
    indicators.loc[index[warmup + 1]:, 'rejection'] = -1.0
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].iloc[warmup + 1:warmup + 5].eq(1).all()
    assert result['signal'].iloc[warmup + 5:].eq(0).all()


if __name__ == '__main__':
    _self_check()
