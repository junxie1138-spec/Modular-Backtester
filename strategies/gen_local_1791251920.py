from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_window: int = 8
    baseline_window: int = 40
    percentile_window: int = 126
    compression_quantile: float = 0.25
    tension_quantile: float = 0.75
    release_quantile: float = 0.65
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791251920'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.compression_window, params.baseline_window) + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (2 <= params.compression_window < params.baseline_window and params.percentile_window >= 2):
            raise ValueError('Require 2 <= compression_window < baseline_window and percentile_window >= 2')
        if not all(0.0 < q < 1.0 for q in (params.compression_quantile, params.tension_quantile, params.release_quantile)):
            raise ValueError('Quantiles must lie strictly between zero and one')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out['return']
        short = params.compression_window
        history = params.percentile_window
        out['short_vol'] = r.rolling(short, min_periods=short).std(ddof=0)
        out['baseline_vol'] = r.rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        out['net_return'] = r.rolling(short, min_periods=short).sum()
        out['gross_return'] = r.abs().rolling(short, min_periods=short).sum()
        # Assess stored tension on the close before the release return.
        out['compression'] = (out['short_vol'] / out['baseline_vol'].replace(0.0, np.nan)).shift(1)
        out['tension'] = (-out['net_return'] / out['gross_return'].replace(0.0, np.nan)).shift(1)
        out['compression_cut'] = out['compression'].rolling(history, min_periods=history).quantile(params.compression_quantile).shift(1)
        out['tension_cut'] = out['tension'].rolling(history, min_periods=history).quantile(params.tension_quantile).shift(1)
        out['release_cut'] = r.rolling(history, min_periods=history).quantile(params.release_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        required = ['return', 'compression', 'tension', 'compression_cut', 'tension_cut', 'release_cut']
        ready = indicators[required].notna().all(axis=1)
        entry = (
            ready
            & (indicators['compression'] <= indicators['compression_cut'])
            & (indicators['tension'] >= indicators['tension_cut'])
            & (indicators['tension'] > 0.0)
            & (indicators['return'] > indicators['release_cut'])
            & (indicators['return'] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    # Force a flat target; entry conditions cannot reset expiry.
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif entry[i]:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        # Both entry and expiry execute one bar after their decision close.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Check exact holding length and next-bar execution under repeated triggers.
    p = StrategyParams()
    index = pd.date_range('2020-01-01', periods=2 * p.holding_bars + 4, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'return': 0.02, 'compression': 0.1, 'tension': 0.8, 'compression_cut': 0.2, 'tension_cut': 0.7, 'release_cut': 0.01}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, p).data
    expected = np.array([0 if i == 0 or (i - 1) % (p.holding_bars + 1) == p.holding_bars else 1 for i in range(len(index))])
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert (result['size'] > 0.0).all()
