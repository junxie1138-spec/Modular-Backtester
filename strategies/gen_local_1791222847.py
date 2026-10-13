from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    block_bars: int = 5
    fast_vol_bars: int = 10
    slow_vol_bars: int = 60
    strain_threshold: float = 1.3
    variance_ceiling: float = 0.7
    shock_threshold: float = 1.1
    profit_target: float = 0.04
    max_hold: int = 18
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.block_bars, self.fast_vol_bars, self.slow_vol_bars) < 2 or self.max_hold < 1:
            raise ValueError('Lookbacks must be at least two and max_hold positive.')
        values = (self.strain_threshold, self.variance_ceiling, self.shock_threshold, self.profit_target, self.min_size, self.max_size)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds and sizes must be finite and positive.')
        if not self.min_size <= self.max_size <= 1.0:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791222847'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.block_bars + params.slow_vol_bars, params.fast_vol_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_return'] = np.log(data['close'].where(data['close'] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        out['block_return'] = out['log_return'].rolling(params.block_bars).sum()
        out['fast_vol'] = out['log_return'].rolling(params.fast_vol_bars).std(ddof=0)
        out['slow_vol'] = out['log_return'].rolling(params.slow_vol_bars).std(ddof=0)
        out['block_vol'] = out['block_return'].rolling(params.slow_vol_bars).std(ddof=0)
        denominator = out['slow_vol'].where(out['slow_vol'] > 0)
        out['variance_ratio'] = out['block_vol'].pow(2) / (params.block_bars * denominator.pow(2))
        out['shock_ratio'] = out['fast_vol'] / denominator
        out['strain'] = out['block_return'] / (np.sqrt(params.block_bars) * denominator)
        out['conviction'] = (
            -out['strain'] / params.strain_threshold - 1.0
            + (params.variance_ceiling - out['variance_ratio']) / params.variance_ceiling
        ).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        valid = np.isfinite(indicators[['strain', 'variance_ratio', 'shock_ratio', 'conviction']]).all(axis=1)
        entries = (
            valid
            & (indicators['strain'] <= -params.strain_threshold)
            & (indicators['variance_ratio'] < params.variance_ceiling)
            & (indicators['shock_ratio'] >= params.shock_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        entry_bar = -1
        entry_price = np.nan
        position_size = params.min_size

        for i in range(len(data)):
            if entry_bar >= 0:
                # Read the entry open only after the entry decision has occurred.
                if i == entry_bar:
                    entry_price = opens[i]
                price_valid = np.isfinite(entry_price) and entry_price > 0
                target_hit = price_valid and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.max_hold
                if not price_valid or target_hit or timed_out:
                    entry_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                entry_price = np.nan
                position_size = params.min_size + (params.max_size - params.min_size) * conviction[i]
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[2], 'close'] = 105.0
    indicators = pd.DataFrame({'strain': -1.4, 'variance_ratio': 0.4, 'shock_ratio': 1.3, 'conviction': 0.3}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=3)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    stronger = indicators.copy()
    stronger['conviction'] = 0.9
    larger = strategy.generate_signals(data, stronger, ctx, params).data
    assert larger['size'].iloc[1] > result['size'].iloc[1] > 0
    assert np.isfinite(result['size']).all()
