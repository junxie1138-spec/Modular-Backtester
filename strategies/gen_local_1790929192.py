from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_window: int = 21
    reference_window: int = 126
    entry_percentile: float = 0.25
    release_percentile: float = 0.60
    drift_window: int = 15
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local_1790929192'

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(params.complexity_window + params.reference_window + 3,
                       params.drift_window + 1, params.volatility_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        windows = (params.complexity_window, params.reference_window,
                   params.drift_window, params.volatility_window, params.atr_window)
        if min(windows) < 2 or params.max_hold < 1:
            raise ValueError('Windows must be at least two and max_hold positive')
        if not 0 < params.entry_percentile < params.release_percentile < 1:
            raise ValueError('Require 0 < entry_percentile < release_percentile < 1')
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.target_volatility, params.breakeven_gain, params.trail_atr)):
            raise ValueError('Risk parameters must be finite and positive')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return'] = returns
        triples = np.column_stack((returns.shift(2), returns.shift(1), returns))
        valid = np.isfinite(triples).all(axis=1)
        # Stable sorting breaks equal-return ties chronologically.
        order = np.argsort(triples, axis=1, kind='stable')
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out['pattern'] = pd.Series(codes, index=data.index, dtype=float).where(valid)
        entropy_terms = []
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = (out['pattern'] == code).astype(float).where(valid)
            frequency = occurrence.rolling(params.complexity_window,
                                           min_periods=params.complexity_window).mean()
            out[f'frequency_{code}'] = frequency
            entropy_terms.append(-frequency * np.log(frequency.clip(lower=1e-12)))
        out['complexity'] = pd.concat(entropy_terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        history = out['complexity'].shift(1).rolling(params.reference_window,
                                                   min_periods=params.reference_window)
        out['entry_threshold'] = history.quantile(params.entry_percentile)
        out['release_threshold'] = history.quantile(params.release_percentile)
        out['drift'] = returns.rolling(params.drift_window,
                                      min_periods=params.drift_window).mean()
        out['volatility'] = returns.rolling(params.volatility_window,
                                           min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        previous = close.shift(1)
        true_range = pd.concat((data['high'] - data['low'],
                                (data['high'] - previous).abs(),
                                (data['low'] - previous).abs()), axis=1).max(axis=1)
        true_range = true_range.where(previous.notna() & data['high'].notna() & data['low'].notna())
        out['atr'] = true_range.rolling(params.atr_window,
                                        min_periods=params.atr_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        volatility = indicators['volatility'].replace(0.0, np.nan)
        exposure = (params.target_volatility / volatility).clip(lower=0.01, upper=0.95)
        df['size'] = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.01).shift(1).fillna(0.01).astype(float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        values = indicators[['complexity', 'entry_threshold', 'release_threshold',
                             'drift', 'volatility', 'atr']].to_numpy(dtype=float)
        active = False
        pending = False
        armed = True
        breakeven = False
        entry = stop = peak = entry_atr = 0.0
        age = 0
        for i in range(len(data)):
            complexity, threshold, release, drift, vol, atr = values[i]
            ready = np.isfinite(values[i]).all() and vol > 0 and atr > 0
            if ready and complexity >= release:
                armed = True
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * entry_atr
                    age = 0
                    breakeven = False
                    active = True
            if active:
                age += 1
                close_valid = np.isfinite(closes[i]) and closes[i] > 0
                # Stops are evaluated at the close and executed next bar.
                if age >= params.max_hold or (close_valid and closes[i] <= stop):
                    active = False
                    continue
                if close_valid:
                    peak = max(peak, closes[i])
                    if closes[i] >= entry * (1.0 + params.breakeven_gain):
                        breakeven = True
                    if breakeven:
                        stop = max(stop, entry)
                        if np.isfinite(atr) and atr > 0:
                            stop = max(stop, peak - params.trail_atr * atr)
                    if closes[i] <= stop:
                        active = False
                        continue
                raw[i] = 1
            elif ready and armed and complexity <= threshold and drift > 0:
                if np.isfinite(closes[i]) and closes[i] > 0:
                    raw[i] = 1
                    pending = True
                    entry_atr = atr
                    armed = False
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({'open': [100.0, 101.0, 104.0, 100.0],
                         'high': [101.0, 105.0, 105.0, 101.0],
                         'low': [99.0, 100.0, 99.0, 99.0],
                         'close': [100.0, 104.0, 100.0, 100.0],
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'complexity': 0.1, 'entry_threshold': 0.2,
                               'release_threshold': 0.8, 'drift': 0.01,
                               'volatility': 0.12, 'atr': 1.0}, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    assert result['size'].gt(0).all()
    unavailable = strategy.generate_signals(data, indicators * np.nan, ctx, params).data
    assert unavailable['signal'].eq(0).all()
    assert strategy.warmup_bars(params) == 150


if __name__ == '__main__':
    _self_check()
