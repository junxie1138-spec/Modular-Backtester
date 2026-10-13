from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    entropy_window: int = 10
    peak_window: int = 40
    atr_window: int = 14
    min_drawdown_atr: float = 1.5
    entropy_floor: float = 0.68
    entropy_gain: float = 0.12
    recoil_atr: float = 0.7
    trail_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.entropy_window, self.peak_window, self.atr_window) < 2 or self.hold_bars < 1:
            raise ValueError('Windows must be at least two and holding period positive')
        values = (self.min_drawdown_atr, self.entropy_gain, self.recoil_atr, self.trail_atr)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Thresholds and ATR multipliers must be finite and positive')
        if not 0 < self.entropy_floor < 1:
            raise ValueError('Entropy floor must lie between zero and one')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791235979'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window + 3, params.atr_window + 4, params.entropy_window + 5))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr = out['atr'].where(out['atr'] > 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['floor'] = close.rolling(5, min_periods=5).min()
        out['drawdown_atr'] = (out['peak'] - close) / atr
        out['recoil_atr'] = (close - out['floor']) / atr

        triples = np.column_stack([close.shift(2), previous, close])
        # Stable ranking assigns earlier observations lower ranks on ties.
        ranks = np.argsort(np.argsort(triples, axis=1, kind='stable'), axis=1, kind='stable')
        codes = ranks @ np.array([9, 3, 1])
        out['ordinal'] = pd.Series(codes, index=data.index, dtype=float).where(np.isfinite(triples).all(axis=1))
        terms = []
        for code in (5, 7, 11, 15, 19, 21):
            name = f'pattern_probability_{code}'
            observations = out['ordinal'].eq(code).astype(float).where(out['ordinal'].notna())
            out[name] = observations.rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            probability = out[name]
            terms.append(-probability * np.log(probability.where(probability > 0, 1.0)))
        out['entropy'] = pd.concat(terms, axis=1).sum(axis=1, skipna=False) / np.log(6.0)
        out['entropy_gain'] = out['entropy'] - out['entropy'].shift(3)
        out['ready'] = (
            (close > out['ma200'])
            & (out['drawdown_atr'].shift(3) >= params.min_drawdown_atr)
            & (out['drawdown_atr'] >= 0.5)
            & (out['recoil_atr'] >= params.recoil_atr)
            & (close > previous)
            & (out['entropy'].shift(3) < params.entropy_floor)
            & (out['entropy'] >= params.entropy_floor)
            & (out['entropy_gain'] >= params.entropy_gain)
            & np.isfinite(atr)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # First update occurs on the fill bar, excluding the decision bar.
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
            elif i >= warmup and ready[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            target[i] = int(active)
        df = pd.DataFrame({'signal': target, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=240, freq='B')
    prices = np.full(240, 100.0)
    prices[205:210] = [150.0, 106.0, 110.0, 108.0, 107.0]
    data = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    indicators = pd.DataFrame({'atr': 1.0, 'ready': False}, index=index)
    indicators.loc[index[205], 'ready'] = True
    indicators.loc[index[208:210], 'atr'] = 10.0
    context = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result['signal'].iloc[205:211].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['signal'].isin([0, 1]).all() and (result['size'] > 0).all()
    indicators['atr'] = 1.0
    flat = data.copy()
    flat['close'] = 100.0
    timed = strategy.generate_signals(flat, indicators, context, params).data['signal']
    assert timed.iloc[206:224].eq(1).all() and timed.iloc[224] == 0
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived['ready'].iloc[:200].any()
    assert np.isfinite(derived['entropy'].dropna()).all()
