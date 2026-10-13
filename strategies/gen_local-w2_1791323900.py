from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRejectionParams:
    peak_lookback: int = 20
    threshold_lookback: int = 126
    deep_percentile: float = 0.85
    shallow_percentile: float = 0.35
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownRejectionParams]):
    strategy_id = 'gen_local-w2_1791323900'

    @classmethod
    def params_type(cls) -> type[DrawdownRejectionParams]:
        return DrawdownRejectionParams

    @staticmethod
    def warmup_bars(params: DrawdownRejectionParams) -> int:
        if params.peak_lookback < 2 or params.threshold_lookback < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0 < params.shallow_percentile < params.deep_percentile < 1:
            raise ValueError('Percentiles must satisfy 0 < shallow < deep < 1.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        return int(params.peak_lookback + params.threshold_lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownRejectionParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['previous_close'] = close.shift(1)
        out['peak'] = out['previous_close'].rolling(
            params.peak_lookback, min_periods=params.peak_lookback
        ).max()
        out['low_depth'] = (
            (out['peak'] - data['low'].where(data['low'] > 0)) / out['peak']
        ).clip(lower=0)
        out['close_depth'] = ((out['peak'] - close) / out['peak']).clip(lower=0)
        out['deep_threshold'] = out['low_depth'].shift(1).rolling(
            params.threshold_lookback, min_periods=params.threshold_lookback
        ).quantile(params.deep_percentile)
        out['shallow_threshold'] = out['close_depth'].shift(1).rolling(
            params.threshold_lookback, min_periods=params.threshold_lookback
        ).quantile(params.shallow_percentile)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['deep_threshold'] > out['shallow_threshold'])
            & (out['low_depth'] > out['deep_threshold'])
            & (out['close_depth'] <= out['shallow_threshold'])
            & (close > out['previous_close'])
            & (close > data['open'])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownRejectionParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                # Force a flat bar even when a new entry condition is present.
                continue
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': True}, index=index)
    for hold in (1, 2):
        result = GeneratedStrategy.generate_signals(
            data, indicators, None, DrawdownRejectionParams(hold_bars=hold)
        ).data
        expected = ([0] + ([1] * hold + [0]) * len(index))[:len(index)]
        assert result['signal'].tolist() == expected
        assert (result['size'] > 0).all()
