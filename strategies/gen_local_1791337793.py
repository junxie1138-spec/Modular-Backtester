from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 10
    threshold_window: int = 126
    smoothing_window: int = 8
    concentration_quantile: float = 0.75
    entry_quantile: float = 0.65
    exit_quantile: float = 0.35


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791337793'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return int(16 + params.energy_window + 2 * params.threshold_window + params.smoothing_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.energy_window, params.threshold_window, params.smoothing_window) < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if not 0 < params.exit_quantile < params.entry_quantile < 1:
            raise ValueError('Require 0 < exit_quantile < entry_quantile < 1.')
        if not 0 < params.concentration_quantile < 1:
            raise ValueError('Concentration quantile must lie between zero and one.')

        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        scales = (2, 4, 8)
        for scale in scales:
            out[f'block_{scale}'] = out['log_close'].rolling(scale, min_periods=scale).sum()
            out[f'detail_{scale}'] = (
                out[f'block_{scale}'] - out[f'block_{scale}'].shift(scale)
            ) / np.sqrt(2.0 * scale)
            out[f'energy_{scale}'] = out[f'detail_{scale}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()

        energies = out[[f'energy_{scale}' for scale in scales]]
        out['total_energy'] = energies.sum(axis=1, min_count=3)
        denominator = out['total_energy'].where(out['total_energy'] > 0)
        out['concentration'] = energies.pow(2).sum(axis=1, min_count=3) / denominator.pow(2)
        out['concentration_cutoff'] = out['concentration'].rolling(
            params.threshold_window, min_periods=params.threshold_window
        ).quantile(params.concentration_quantile).shift(1)

        weighted_details = pd.DataFrame(index=data.index)
        for scale in scales:
            out[f'weight_{scale}'] = out[f'energy_{scale}'] / denominator
            out[f'weighted_detail_{scale}'] = out[f'weight_{scale}'] * out[f'detail_{scale}']
            weighted_details[scale] = out[f'weighted_detail_{scale}']
        out['projection'] = weighted_details.sum(axis=1, min_count=3) / np.sqrt(denominator)
        ready = out['concentration'].notna() & out['concentration_cutoff'].notna()
        out['reversal_regime'] = (
            out['concentration'] > out['concentration_cutoff']
        ).astype(float).where(ready)
        out['regime_score'] = out['projection'] * (1.0 - 2.0 * out['reversal_regime'])
        out['score'] = out['regime_score'].rolling(
            params.smoothing_window, min_periods=params.smoothing_window
        ).mean()
        out['entry_threshold'] = out['score'].rolling(
            params.threshold_window, min_periods=params.threshold_window
        ).quantile(params.entry_quantile).shift(1)
        out['exit_threshold'] = out['score'].rolling(
            params.threshold_window, min_periods=params.threshold_window
        ).quantile(params.exit_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        # The bullish condition is latched between its historical percentile boundaries.
        # Its reversal is the sole exit; missing observations preserve its current state.
        values = indicators[['score', 'entry_threshold', 'exit_threshold']].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        bullish = False
        for bar, (score, upper, lower) in enumerate(values):
            if np.isfinite(score) and np.isfinite(upper) and np.isfinite(lower):
                if not bullish and score > upper:
                    bullish = True
                elif bullish and score < lower:
                    bullish = False
            signals[bar] = int(bullish)

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = WaveletParams()
    count = 800
    time = np.arange(count, dtype=float)
    close = 100.0 * np.exp(0.0002 * time + 0.025 * np.sin(time / 13.0) + 0.009 * np.sin(time / 3.0))
    data = pd.DataFrame(
        {'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0},
        index=pd.date_range('2020-01-01', periods=count, freq='B'),
    )
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result.index.equals(data.index)
    assert result['signal'].isin([0, 1]).all()
    assert result['signal'].dtype.kind in 'iu'
    assert (result['size'] > 0).all()
    assert result['signal'].iloc[:GeneratedStrategy.warmup_bars(params) - 2].eq(0).all()
    prefix = data.iloc[:600]
    prefix_indicators = GeneratedStrategy.indicators(prefix, params)
    prefix_result = GeneratedStrategy.generate_signals(prefix, prefix_indicators, None, params).data
    assert result.iloc[:600].equals(prefix_result)

    probe = pd.DataFrame(
        {'score': [2.0, 0.0, -2.0, 0.0], 'entry_threshold': 1.0, 'exit_threshold': -1.0},
        index=data.index[:4],
    )
    shifted = GeneratedStrategy.generate_signals(data.iloc[:4], probe, None, params).data
    assert shifted['signal'].tolist() == [0, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
