from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    smooth_bars: int = 15
    percentile_bars: int = 126
    entry_percentile: float = 0.60

    def __post_init__(self):
        if self.smooth_bars < 1 or self.percentile_bars < 2:
            raise ValueError('Lookback windows must be positive; percentile_bars must be at least two.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one.')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791245466'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(32 + params.smooth_bars + params.percentile_bars)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        energy_names = []
        signed_names = []
        for half in (2, 4, 8, 16):
            recent = f'recent_{half}'
            previous = f'previous_{half}'
            coefficient = f'haar_{half}'
            energy = f'energy_{half}'
            signed = f'signed_energy_{half}'
            out[recent] = out['log_close'].rolling(half, min_periods=half).mean()
            out[previous] = out[recent].shift(half)
            out[coefficient] = np.sqrt(half / 2.0) * (out[recent] - out[previous])
            out[energy] = out[coefficient].pow(2)
            out[signed] = np.sign(out[coefficient]) * out[energy]
            energy_names.append(energy)
            signed_names.append(signed)
        out['total_energy'] = out[energy_names].sum(axis=1, min_count=4)
        out['signed_energy'] = out[signed_names].sum(axis=1, min_count=4)
        out['squared_energy_sum'] = out[energy_names].pow(2).sum(axis=1, min_count=4)
        total = out['total_energy'].replace(0.0, np.nan)
        squares = out['squared_energy_sum'].replace(0.0, np.nan)
        out['directional_agreement'] = out['signed_energy'] / total
        out['energy_breadth'] = out['total_energy'].pow(2) / (4.0 * squares)
        out['raw_strength'] = out['directional_agreement'] * out['energy_breadth']
        out.loc[out['total_energy'].eq(0.0), 'raw_strength'] = 0.0
        out['strength'] = out['raw_strength'].rolling(
            params.smooth_bars, min_periods=params.smooth_bars
        ).mean()
        out['entry_threshold'] = out['strength'].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        strength = indicators['strength']
        threshold = indicators['entry_threshold']
        ready = strength.notna() & threshold.notna()
        # The position exits exactly when the same entry condition becomes false.
        entry = ready & strength.gt(threshold) & strength.gt(0.0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = entry.astype(int)
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = WaveletParams()
    strategy = GeneratedStrategy()
    rng = np.random.default_rng(23)
    index = pd.date_range('2000-01-01', periods=600, freq='B')
    returns = 0.002 * np.sin(np.arange(600) / 25.0) + rng.normal(0.0, 0.003, 600)
    close = 100.0 * np.exp(np.cumsum(returns))
    data = pd.DataFrame({
        'open': close, 'high': close * 1.005, 'low': close * 0.995,
        'close': close, 'volume': 1000000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].isin([0, 1]).all()
    assert signals['signal'].iloc[0] == 0
    assert np.isfinite(signals['size']).all() and signals['size'].gt(0).all()
    assert signals['signal'].eq(1).any()
    assert signals['signal'].diff().eq(-1).any()
    assert signals['signal'].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()
    prefix = data.iloc[:400]
    prefix_indicators = strategy.indicators(prefix, params)
    prefix_signals = strategy.generate_signals(prefix, prefix_indicators, None, params).data
    pd.testing.assert_frame_equal(indicators.iloc[:400], prefix_indicators)
    pd.testing.assert_frame_equal(signals.iloc[:400], prefix_signals)
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    flat_indicators = strategy.indicators(flat, params)
    flat_signals = strategy.generate_signals(flat, flat_indicators, None, params).data
    assert flat_signals['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
