from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 3
    vol_window: int = 20
    atr_bars: int = 14
    analog_bars: int = 252
    bandwidth: float = 0.8
    minimum_edge: float = 0.25
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791265072'

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(params.analog_bars + max(params.roc_bars + params.vol_window + 2, params.atr_bars) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if min(params.roc_bars, params.vol_window, params.atr_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if params.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five')
        if params.analog_bars < params.hold_bars + 8:
            raise ValueError('Insufficient analog history')
        if params.bandwidth <= 0 or params.minimum_edge <= 0 or params.stop_atr <= 0:
            raise ValueError('Thresholds must be positive')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['acceleration_vol'] = out['acceleration'].rolling(params.vol_window).std().replace(0.0, np.nan)
        out['a0'] = out['acceleration'] / out['acceleration_vol']
        out['a1'] = out['a0'].shift(1)
        out['a2'] = out['a0'].shift(2)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean().replace(0.0, np.nan)
        out['roc_atr'] = out['roc'] / (out['atr'] / close.replace(0.0, np.nan))
        weights = pd.Series(0.0, index=data.index)
        weighted_returns = weights.copy()
        weighted_wins = weights.copy()
        squared_weights = weights.copy()
        features = out[['a0', 'a1', 'a2', 'roc_atr']]
        # ponytail: bounded lag scan; batch distances if history grows materially.
        for lag in range(params.hold_bars + 1, params.analog_bars + 1):
            historical = features.shift(lag)
            distance = (features - historical).pow(2).mean(axis=1, skipna=False)
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2))
            outcome = (close.shift(lag - params.hold_bars) - data['open'].shift(lag - 1)) / out['atr'].shift(lag)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            weight = weight.where(valid, 0.0).fillna(0.0)
            outcome = outcome.where(valid, 0.0).fillna(0.0)
            weights += weight
            squared_weights += weight.pow(2)
            weighted_returns += weight * outcome
            weighted_wins += weight * (outcome > 0).astype(float)
        denominator = weights.replace(0.0, np.nan)
        out['analog_weight'] = weights
        out['edge'] = weighted_returns / denominator
        out['win_share'] = weighted_wins / denominator
        out['effective_matches'] = weights.pow(2) / squared_weights.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        eligible = (
            (indicators['roc'] < 0)
            & (indicators['a0'] > 0)
            & (indicators['a0'] > indicators['a1'])
            & (indicators['edge'] >= params.minimum_edge)
            & (indicators['win_share'] >= 0.58)
            & (indicators['effective_matches'] >= 8.0)
            & (indicators['analog_weight'] >= 2.0)
            & (indicators['atr'] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        conviction = (
            (indicators['edge'] / (2.0 * params.minimum_edge)).clip(0.0, 1.0)
            * ((indicators['win_share'] - 0.5) / 0.25).clip(0.0, 1.0)
        ).fillna(0.0)
        entry_sizes = (0.20 + 0.75 * conviction).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.20, dtype=float)
        pending = False
        active = False
        entry_bar = 0
        entry_atr = 0.0
        stop = 0.0
        held_size = 0.20
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * entry_atr
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    active = False
                    continue
                signal[i] = 1
                size[i] = held_size
            elif i >= warmup and eligible[i] and i + 1 < len(data):
                pending = True
                entry_atr = atr[i]
                held_size = entry_sizes[i]
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AnalogParams(roc_bars=2, vol_window=3, atr_bars=3, analog_bars=12)
    start = GeneratedStrategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=start + 12, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'roc': -0.01, 'a0': 1.0, 'a1': 0.0, 'edge': 0.0, 'win_share': 0.75, 'effective_matches': 10.0, 'analog_weight': 3.0, 'atr': 1.0}, index=index)
    features.loc[index[start], 'edge'] = 1.0
    for stopped in (False, True):
        sample = data.copy()
        if stopped:
            sample.loc[index[start + 2], 'close'] = 98.0
        result = GeneratedStrategy.generate_signals(sample, features, None, params).data
        exit_bar = start + 3 if stopped else start + params.hold_bars + 1
        assert result['signal'].iloc[start] == 0
        assert (result['signal'].iloc[start + 1:exit_bar] == 1).all()
        assert result['signal'].iloc[exit_bar] == 0
        assert result['signal'].isin([0, 1]).all()
        assert (result['size'] > 0).all()
        assert result['size'].iloc[start + 1] > 0.20
    t = np.arange(120, dtype=float)
    index = pd.date_range('2021-01-01', periods=len(t), freq='B')
    close = 100.0 + 0.03 * t + np.sin(t / 3.0)
    sample = pd.DataFrame({'open': close + 0.1, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
    full = GeneratedStrategy.indicators(sample, params)
    prefix = GeneratedStrategy.indicators(sample.iloc[:90], params)
    assert np.allclose(full.iloc[:90].to_numpy(), prefix.to_numpy(), equal_nan=True)


if __name__ == '__main__':
    _self_check()
