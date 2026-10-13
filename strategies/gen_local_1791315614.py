from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    trend_window: int = 20
    innovation_window: int = 40
    volatility_window: int = 20
    atr_window: int = 14
    entry_threshold: float = 0.5
    target_volatility: float = 0.12
    atr_multiple: float = 3.0
    holding_bars: int = 18
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.trend_window, self.innovation_window, self.volatility_window, self.atr_window) < 3:
            raise ValueError('Indicator windows must be at least three bars.')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive.')
        if not all(np.isfinite(v) and v > 0 for v in (self.entry_threshold, self.target_volatility, self.atr_multiple, self.max_size)):
            raise ValueError('Thresholds and sizing parameters must be finite and positive.')
        if self.max_size > 1:
            raise ValueError('max_size must not exceed one.')


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = 'gen_local_1791315614'

    @classmethod
    def params_type(cls):
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return int(max(params.trend_window + 2 * params.innovation_window + 3,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        n = params.trend_window
        w = params.innovation_window
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        out['time'] = np.arange(len(data), dtype=float)
        out['level_mean'] = out['log_close'].rolling(n, min_periods=n).mean()
        out['slope'] = out['log_close'].rolling(n, min_periods=n).cov(out['time'], ddof=0) / ((n * n - 1) / 12.0)
        out['latent_level'] = out['level_mean'] + out['slope'] * ((n - 1) / 2.0)
        out['level_innovation'] = out['log_close'] - out['latent_level'].shift(1) - out['slope'].shift(1)
        out['slope_innovation'] = out['slope'].diff()
        out['level_variance'] = out['level_innovation'].rolling(w, min_periods=w).var(ddof=0).shift(1)
        out['innovation_covariance'] = out['slope_innovation'].rolling(w, min_periods=w).cov(out['level_innovation'], ddof=0).shift(1)
        out['beta'] = out['innovation_covariance'] / out['level_variance'].where(out['level_variance'] > 1e-16)
        out['prior_slope_mean'] = out['slope_innovation'].rolling(w, min_periods=w).mean().shift(1)
        out['prior_level_mean'] = out['level_innovation'].rolling(w, min_periods=w).mean().shift(1)
        out['orthogonal_innovation'] = (out['slope_innovation'] - out['prior_slope_mean']
                                        - out['beta'] * (out['level_innovation'] - out['prior_level_mean']))
        out['innovation_scale'] = np.sqrt(out['orthogonal_innovation'].pow(2).rolling(w, min_periods=w).mean().shift(1))
        out['innovation_score'] = out['orthogonal_innovation'] / out['innovation_scale'].where(out['innovation_scale'] > 1e-12)
        out['return'] = out['log_close'].diff()
        out['annual_volatility'] = out['return'].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(),
                                      (data['high'] - previous_close).abs(),
                                      (data['low'] - previous_close).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['exposure'] = (params.target_volatility / out['annual_volatility'].where(out['annual_volatility'] > 1e-12)).clip(lower=0.01, upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: InnovationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        required = indicators[['slope', 'innovation_score', 'atr', 'annual_volatility', 'exposure']]
        valid = pd.Series(np.isfinite(required.to_numpy()).all(axis=1), index=data.index)
        valid &= np.isfinite(data['close']) & (data['close'] > 0)
        valid &= (indicators['atr'] > 0) & (indicators['annual_volatility'] > 0) & (indicators['exposure'] > 0)
        long_candidate = valid & (indicators['slope'] > 0) & (indicators['innovation_score'] >= params.entry_threshold)
        short_candidate = valid & (indicators['slope'] < 0) & (indicators['innovation_score'] <= -params.entry_threshold)
        confirmed_long = long_candidate & long_candidate.shift(1, fill_value=False)
        confirmed_short = short_candidate & short_candidate.shift(1, fill_value=False)
        entries = np.where(confirmed_long, 1, np.where(confirmed_short, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        valid_values = valid.to_numpy(dtype=bool)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                age += 1
                if not valid_values[i]:
                    position = 0
                else:
                    # First active close occurs after the entry decision's next-bar fill.
                    if position == 1:
                        extreme = closes[i] if np.isnan(extreme) else max(extreme, closes[i])
                        candidate_stop = extreme - params.atr_multiple * atrs[i]
                        stop = candidate_stop if np.isnan(stop) else max(stop, candidate_stop)
                        breached = closes[i] <= stop
                    else:
                        extreme = closes[i] if np.isnan(extreme) else min(extreme, closes[i])
                        candidate_stop = extreme + params.atr_multiple * atrs[i]
                        stop = candidate_stop if np.isnan(stop) else min(stop, candidate_stop)
                        breached = closes[i] >= stop
                    if breached or age >= params.holding_bars:
                        position = 0
                # An exit remains flat for this decision bar; no immediate reentry.
            elif valid_values[i] and entries[i]:
                position = int(entries[i])
                age = 0
                extreme = np.nan
                stop = np.nan
            signals[i] = position
        df['signal'] = signals
        df['size'] = indicators['exposure'].shift(1).fillna(min(0.01, params.max_size)).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = InnovationParams()
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    for direction, closes in ((1, [100, 100, 101, 110, 105, 104]),
                              (-1, [100, 100, 99, 90, 95, 96])):
        data = pd.DataFrame({'close': closes}, index=index)
        indicators = pd.DataFrame({'slope': float(direction),
                                   'innovation_score': float(direction),
                                   'atr': [1, 1, 1, 1, 10, 10],
                                   'annual_volatility': 0.2,
                                   'exposure': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, 0]
        assert np.isclose(result['size'].iloc[2], 0.5)
        assert (result['size'] > 0).all()
        indicators['innovation_score'] = [direction, 0, direction, 0, direction, 0]
        assert not strategy.generate_signals(data, indicators, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
