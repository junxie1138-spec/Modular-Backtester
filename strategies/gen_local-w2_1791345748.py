from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 18
    horizon_bars: int = 18
    training_bars: int = 504
    smoothing_bars: int = 3
    min_month_samples: int = 20
    entry_threshold: float = 0.002


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = 'gen_local-w2_1791345748'

    @classmethod
    def params_type(cls) -> type[SignatureSeasonParams]:
        return SignatureSeasonParams

    @staticmethod
    def warmup_bars(params: SignatureSeasonParams) -> int:
        return int(params.path_bars + params.horizon_bars + params.training_bars + params.smoothing_bars + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        if min(params.path_bars, params.horizon_bars, params.training_bars, params.smoothing_bars, params.min_month_samples) < 1:
            raise ValueError('Lookbacks and sample counts must be positive.')
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError('entry_threshold must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        length = params.path_bars
        out['month'] = data.index.month
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        out['log_return'] = out['log_close'].diff()
        out['path_volatility'] = out['log_return'].rolling(length, min_periods=length).std(ddof=0)
        out['path_start'] = out['log_close'].shift(length)
        out['endpoint_mean'] = 0.5 * (out['log_close'] + out['path_start'])
        # Antisymmetric level-two signature for normalized time and log price:
        # one half of integral(time dprice - price dtime), after translation.
        out['path_integral'] = (out['log_close'].rolling(length + 1, min_periods=length + 1).sum() - out['endpoint_mean']) / length
        out['signature_area'] = out['endpoint_mean'] - out['path_integral']
        out['normalized_area'] = out['signature_area'].div((out['path_volatility'] * np.sqrt(length)).replace(0.0, np.nan)).clip(-4.0, 4.0)
        # Training pairs become available only after their entire future return has realized.
        out['mature_area'] = out['normalized_area'].shift(params.horizon_bars)
        out['mature_return'] = out['log_close'].diff(params.horizon_bars)
        out['origin_month'] = out['month'].shift(params.horizon_bars)
        out['area_return_product'] = out['mature_area'] * out['mature_return']
        out['area_square'] = out['mature_area'].pow(2)
        out['seasonal_coefficient'] = np.nan
        valid = out['mature_area'].notna() & out['mature_return'].notna()
        for month in range(1, 13):
            eligible = valid & out['origin_month'].eq(month)
            prefix = f'month_{month}_'
            out[prefix + 'count'] = eligible.astype(float).rolling(params.training_bars, min_periods=params.training_bars).sum()
            out[prefix + 'cross'] = out['area_return_product'].where(eligible, 0.0).rolling(params.training_bars, min_periods=params.training_bars).sum()
            out[prefix + 'square'] = out['area_square'].where(eligible, 0.0).rolling(params.training_bars, min_periods=params.training_bars).sum()
            out[prefix + 'coefficient'] = out[prefix + 'cross'].div(out[prefix + 'square'].where(out[prefix + 'square'] > 1e-12)).where(out[prefix + 'count'] >= params.min_month_samples)
            current = out['month'].eq(month)
            out.loc[current, 'seasonal_coefficient'] = out.loc[current, prefix + 'coefficient']
        out['raw_forecast'] = out['seasonal_coefficient'] * out['normalized_area']
        out['forecast'] = out['raw_forecast'].rolling(params.smoothing_bars, min_periods=params.smoothing_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        forecast = indicators['forecast'].replace([np.inf, -np.inf], np.nan)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[forecast >= params.entry_threshold] = 1
        candidate.loc[forecast <= -params.entry_threshold] = -1
        candidate.iloc[:self.warmup_bars(params) - 1] = 0
        confirmed = candidate.ne(0) & candidate.eq(candidate.shift(1))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate.to_numpy()):
            if position != 0:
                # An opposite entry condition exits immediately; neutral bars hold.
                # Re-entry requires confirmation and cannot occur on the exit bar.
                if direction == -position:
                    position = 0
            elif confirmed.iloc[i]:
                position = int(direction)
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Runnable check: confirmation, neutral holding, reversal exit, and fill delay.
    params = SignatureSeasonParams(path_bars=1, horizon_bars=1, training_bars=1, smoothing_bars=1, min_month_samples=1)
    strategy = GeneratedStrategy()
    padding = strategy.warmup_bars(params)
    sequence = [0.0, 0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.01, 0.0]
    index = pd.date_range('2020-01-01', periods=padding + len(sequence), freq='B')
    data = pd.DataFrame({name: 100.0 for name in ['open', 'high', 'low', 'close', 'volume']}, index=index)
    features = pd.DataFrame({'forecast': [0.0] * padding + sequence}, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].iloc[padding:].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, 0, 1]
    assert result['signal'].isin([-1, 0, 1]).all() and (result['size'] > 0).all()
