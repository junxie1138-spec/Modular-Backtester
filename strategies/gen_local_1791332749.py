from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ThroughputParams:
    lookback: int = 8
    throughput_threshold: float = 0.45
    pressure_threshold: float = 0.12

    def __post_init__(self):
        if not 2 <= self.lookback <= 8:
            raise ValueError('lookback must be between 2 and 8')
        if not 0.0 < self.throughput_threshold < 1.0:
            raise ValueError('throughput_threshold must be between 0 and 1')
        if not 0.0 < self.pressure_threshold < 1.0:
            raise ValueError('pressure_threshold must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[ThroughputParams]):
    strategy_id = 'gen_local_1791332749'

    @classmethod
    def params_type(cls) -> type[ThroughputParams]:
        return ThroughputParams

    def warmup_bars(self, params: ThroughputParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: ThroughputParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prices = data[['open', 'high', 'low', 'close']].replace([np.inf, -np.inf], np.nan)
        out['valid'] = prices.notna().all(axis=1) & prices['high'].ge(prices['low'])
        out['range'] = (prices['high'] - prices['low']).where(out['valid']).replace(0.0, np.nan)
        out['volume'] = data['volume'].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
        out['volume_baseline'] = out['volume'].rolling(2, min_periods=2).mean().shift(1)
        out['body'] = ((prices['close'] - prices['open']) / out['range']).clip(-1.0, 1.0).fillna(0.0)
        out['location'] = ((2.0 * prices['close'] - prices['high'] - prices['low']) / out['range']).clip(-1.0, 1.0).fillna(0.0)
        out['confirmed'] = out['valid'] & out['volume'].gt(0.0) & out['volume_baseline'].gt(0.0) & out['volume'].ge(out['volume_baseline'])
        out['confirmed_volume'] = out['volume'].where(out['confirmed'], 0.0)
        out['body_weight'] = out['confirmed_volume'] * out['body']
        out['location_weight'] = out['confirmed_volume'] * out['location']
        out['throughput_weight'] = out['confirmed_volume'] * out['body'].abs()
        window = params.lookback
        out['confirmed_total'] = out['confirmed_volume'].rolling(window, min_periods=window).sum()
        out['body_total'] = out['body_weight'].rolling(window, min_periods=window).sum()
        out['location_total'] = out['location_weight'].rolling(window, min_periods=window).sum()
        out['throughput_total'] = out['throughput_weight'].rolling(window, min_periods=window).sum()
        denominator = out['confirmed_total'].replace(0.0, np.nan)
        out['body_pressure'] = (out['body_total'] / denominator).fillna(0.0)
        out['absorption_pressure'] = (out['location_total'] / denominator).fillna(0.0)
        out['throughput'] = (out['throughput_total'] / denominator).fillna(0.0)
        out['ready'] = out['valid'].rolling(window, min_periods=window).sum().eq(window) & out['volume'].notna().rolling(window + 2, min_periods=window + 2).sum().eq(window + 2) & out['confirmed_total'].gt(0.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ThroughputParams) -> SignalFrame:
        trending = indicators['throughput'].ge(params.throughput_threshold)
        pressure = indicators['body_pressure'].where(trending, indicators['absorption_pressure'])
        entry_condition = indicators['ready'].astype(bool) & pressure.gt(params.pressure_threshold)
        df = pd.DataFrame(index=data.index)
        # The position persists exactly while the regime-specific entry condition holds.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ThroughputParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'throughput': [0.8, 0.8, 0.2, 0.2, 0.8, 0.8], 'body_pressure': [0.3, 0.3, -0.3, -0.3, -0.3, 0.3], 'absorption_pressure': [-0.3, -0.3, 0.3, -0.3, 0.3, 0.3], 'ready': [False, True, True, True, True, True]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    assert result['size'].gt(0.0).all()
    assert strategy.warmup_bars(params) == 10
    index = pd.date_range('2020-01-01', periods=20)
    data = data.iloc[:1].reindex(index).ffill()
    derived = strategy.indicators(data, params)
    assert np.isfinite(derived.to_numpy(dtype=float)).all()
    assert not derived['ready'].iloc[:9].any()
    prefix = strategy.indicators(data.iloc[:15], params)
    assert np.allclose(derived.iloc[:15].to_numpy(dtype=float), prefix.to_numpy(dtype=float))


if __name__ == '__main__':
    _self_check()
