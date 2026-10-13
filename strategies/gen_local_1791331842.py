from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    topology_lags: int = 8
    calibration_bars: int = 40
    correlation_threshold: float = 0.20
    holding_bars: int = 2

    def __post_init__(self):
        if self.topology_lags < 2 or self.calibration_bars < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError('Correlation threshold must be in (0, 1].')
        if self.holding_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars.')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791331842'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(200, params.topology_lags + params.calibration_bars + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return'] = close.div(close.shift(1).replace(0.0, np.nan)).sub(1.0).replace([np.inf, -np.inf], np.nan)
        out['magnitude'] = out['return'].abs()
        out['sign'] = np.sign(out['return'])
        out['edge_count'] = 0.0
        out['sign_agreement'] = 0.0
        magnitude = out['magnitude']
        signs = out['sign']
        interior_max = magnitude.shift(1)
        for lag in range(2, params.topology_lags + 1):
            if lag > 2:
                interior_max = pd.concat([interior_max, magnitude.shift(lag - 1)], axis=1).max(axis=1, skipna=False)
            endpoint_min = pd.concat([magnitude, magnitude.shift(lag)], axis=1).min(axis=1, skipna=False)
            edge = interior_max.lt(endpoint_min) & endpoint_min.notna() & interior_max.notna()
            out[f'edge_{lag}'] = edge.astype(float)
            out['edge_count'] += out[f'edge_{lag}']
            out['sign_agreement'] += signs.mul(signs.shift(lag)).where(edge, 0.0).fillna(0.0)
        out['graph_ready'] = out['return'].notna().rolling(params.topology_lags + 1, min_periods=params.topology_lags + 1).sum().eq(params.topology_lags + 1)
        count = out['edge_count'].where(out['graph_ready'])
        agreement = out['sign_agreement'].where(out['graph_ready'])
        out['calibration_edges'] = count.rolling(params.calibration_bars, min_periods=params.calibration_bars).sum().shift(1)
        out['calibration_agreement'] = agreement.rolling(params.calibration_bars, min_periods=params.calibration_bars).sum().shift(1)
        out['edge_autocorrelation'] = out['calibration_agreement'].div(out['calibration_edges'].replace(0.0, np.nan))
        out['forecast'] = out['sign'].mul(out['edge_autocorrelation'])
        out['ready'] = out['graph_ready'] & out['ma200'].notna() & out['forecast'].notna()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        valid = indicators['ready'].fillna(False) & indicators['edge_count'].ge(1.0)
        long_entry = valid & indicators['forecast'].ge(params.correlation_threshold) & data['close'].gt(indicators['ma200'])
        short_entry = valid & indicators['forecast'].le(-params.correlation_threshold) & data['close'].lt(indicators['ma200'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                else:
                    signal[bar] = position
                continue
            if entry != 0:
                position = int(entry)
                exit_bar = bar + params.holding_bars
                signal[bar] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [101, 99, 99, 99, 101, 101, 101, 101]}, index=index)
    indicators = pd.DataFrame({'ready': True, 'edge_count': 1.0, 'ma200': 100.0, 'forecast': [1, -1, -1, -1, 1, 1, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, VisibilityParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, VisibilityParams(holding_bars=1))
    assert result.data['signal'].tolist() == [0, 1, 0, -1, 0, 1, 0, 1]
