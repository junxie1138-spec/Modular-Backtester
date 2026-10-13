from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_bars: int = 5
    percentile_window: int = 80
    compression_quantile: float = 0.30
    divergence_quantile: float = 0.80
    profit_target: float = 0.006
    max_hold: int = 2

    def __post_init__(self):
        if self.compression_bars < 2 or self.percentile_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not (0.0 < self.compression_quantile < 1.0 and 0.0 < self.divergence_quantile < 1.0):
            raise ValueError('Quantiles must lie strictly between zero and one.')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('Profit target must be finite and positive.')
        if self.max_hold not in (1, 2):
            raise ValueError('Holding period must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1790982265'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.compression_bars + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        b = params.compression_bars
        w = params.percentile_window
        close = data['close'].where(np.isfinite(data['close']) & (data['close'] > 0.0))
        volume = data['volume'].where(np.isfinite(data['volume']) & (data['volume'] >= 0.0))
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['absolute_return'] = out['return'].abs()
        out['volume_return'] = out['return'] * volume
        out['volume_sum'] = volume.rolling(b, min_periods=b).sum()
        out['volume_return_sum'] = out['volume_return'].rolling(b, min_periods=b).sum()
        out['mean_return'] = out['return'].rolling(b, min_periods=b).mean()
        out['weighted_return'] = out['volume_return_sum'] / out['volume_sum'].where(out['volume_sum'] > 0.0)
        out['compression'] = out['absolute_return'].rolling(b, min_periods=b).mean()
        out['divergence'] = (out['weighted_return'] - out['mean_return']).abs()
        out['compression_threshold'] = out['compression'].rolling(w, min_periods=w).quantile(params.compression_quantile).shift(1)
        out['divergence_threshold'] = out['divergence'].rolling(w, min_periods=w).quantile(params.divergence_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        columns = ['mean_return', 'weighted_return', 'compression', 'divergence', 'compression_threshold', 'divergence_threshold']
        ready = np.isfinite(indicators[columns]).all(axis=1)
        qualifies = (
            ready
            & (indicators['compression'] <= indicators['compression_threshold'])
            & (indicators['divergence'] > indicators['divergence_threshold'])
            & (indicators['mean_return'] * indicators['weighted_return'] < 0.0)
        )
        entries = np.where(qualifies, np.sign(indicators['weighted_return']), 0).astype(int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        decision_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position != 0:
                # Observe the next-session entry open only when that bar arrives.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                timed_out = i - decision_bar >= params.max_hold
                if target_hit or timed_out or not valid_price:
                    position = 0
                    entry_price = np.nan
            elif entries[i] != 0:
                position = int(entries[i])
                decision_bar = i
                entry_price = np.nan
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': [100.0, 101.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'mean_return': -0.001, 'weighted_return': 0.001, 'compression': 0.0005, 'compression_threshold': 0.001, 'divergence': 0.002, 'divergence_threshold': 0.001}, index=index)
    ind.loc[index[2], ['mean_return', 'weighted_return']] = [0.001, -0.001]
    ind.loc[index[7], 'divergence'] = 0.0
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 0, -1, 0, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    warmup_result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert warmup_result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
