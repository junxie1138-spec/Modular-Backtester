from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 16
    correlation_window: int = 48
    smoothing_window: int = 16
    percentile_window: int = 126
    entry_quantile: float = 0.65


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791318847'

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return int(33 + max(params.energy_window, params.correlation_window) + params.smoothing_window + params.percentile_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.energy_window, params.correlation_window, params.smoothing_window, params.percentile_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError('entry_quantile must lie between zero and one.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(lambda x: x > 0)
        out['log_return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        for scale in (2, 8, 16):
            half = scale // 2
            block = out['log_return'].rolling(half, min_periods=half).sum()
            out[f'block_{scale}'] = block
            out[f'haar_{scale}'] = (block - block.shift(half)) / np.sqrt(float(scale))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out['total_energy'] = out[['energy_2', 'energy_8', 'energy_16']].sum(axis=1, min_count=3)
        out['slow_share'] = out['energy_16'].div(out['total_energy'].where(out['total_energy'] > 0))
        out['slow_lag'] = out['haar_16'].shift(16)
        out['persistence'] = out['haar_16'].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out['slow_lag']).clip(-1.0, 1.0)
        out['phase'] = out['log_return'].rolling(16, min_periods=16).sum()
        out['raw_score'] = out['slow_share'] * out['persistence'].clip(lower=0.0) * np.sign(out['phase'])
        out['score'] = out['raw_score'].rolling(params.smoothing_window, min_periods=params.smoothing_window).mean()
        out['threshold'] = out['score'].abs().shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        ready = indicators[['score', 'threshold']].notna().all(axis=1)
        active = ready & indicators['score'].gt(indicators['threshold'])
        # The entry predicate defines the held regime; its reversal exits.
        df = pd.DataFrame(index=data.index)
        df['signal'] = active.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = WaveletParams()
    rng = np.random.default_rng(7)
    n = GeneratedStrategy.warmup_bars(params) + 600
    close = 100.0 * np.exp(np.cumsum(0.0004 + rng.normal(0.0, 0.01, n)))
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0}, index=pd.date_range('2020-01-01', periods=n, freq='B'))
    ind = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    expected = (ind[['score', 'threshold']].notna().all(axis=1) & ind['score'].gt(ind['threshold'])).astype(int).shift(1).fillna(0).astype(int)
    assert result['signal'].equals(expected)
    assert result.index.equals(data.index)
    assert result['signal'].isin([0, 1]).all() and result['size'].gt(0).all()
    assert ind[['score', 'threshold']].iloc[GeneratedStrategy.warmup_bars(params)].notna().all()
    prefix = data.iloc[:-50]
    prefix_ind = GeneratedStrategy.indicators(prefix, params)
    prefix_result = GeneratedStrategy.generate_signals(prefix, prefix_ind, None, params).data
    pd.testing.assert_frame_equal(ind.iloc[:-50], prefix_ind)
    pd.testing.assert_frame_equal(result.iloc[:-50], prefix_result)
