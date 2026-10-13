from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    slope_window: int = 8
    correlation_window: int = 40
    percentile_window: int = 126
    echo_percentile: float = 0.85
    regime_percentile: float = 0.65
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local_1791254536'

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.atr_window + 1, params.slope_window + params.correlation_window + params.percentile_window + 7))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.slope_window, params.correlation_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not (0.0 < params.echo_percentile < 1.0 and 0.0 < params.regime_percentile < 1.0):
            raise ValueError('Percentiles must lie strictly between zero and one.')
        if params.trail_atr <= 0.0 or params.max_hold < 1:
            raise ValueError('Trailing distance and holding limit must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['log_close'] = np.log(close)
        out['return'] = out['log_close'].diff()
        out['slope'] = out['return'].rolling(params.slope_window).mean()
        out['prior_slope'] = out['slope'].shift(1)
        out['innovation'] = out['return'] - out['prior_slope']
        out['innovation_lag1'] = out['innovation'].shift(1)
        out['innovation_lag5'] = out['innovation'].shift(5)
        out['daily_corr'] = out['innovation'].rolling(params.correlation_window).corr(out['innovation_lag1'])
        out['weekly_corr'] = out['innovation'].rolling(params.correlation_window).corr(out['innovation_lag5'])
        out['weekly_corr'] = out['weekly_corr'].clip(-1.0, 1.0)
        out['daily_corr'] = out['daily_corr'].clip(-1.0, 1.0)
        out['phase_predecessor'] = out['innovation'].shift(4)
        out['echo'] = out['weekly_corr'] * out['phase_predecessor']
        out['predicted_slope'] = out['slope'] + out['echo'] / params.slope_window
        out['echo_magnitude'] = out['echo'].abs()
        out['echo_threshold'] = out['echo_magnitude'].shift(1).rolling(params.percentile_window).quantile(params.echo_percentile)
        out['regime_threshold'] = out['weekly_corr'].shift(1).rolling(params.percentile_window).quantile(params.regime_percentile)
        out['previous_close'] = close.shift(1)
        out['range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['range', 'high_gap', 'low_gap']].max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['weekly_corr', 'daily_corr', 'echo', 'predicted_slope', 'echo_threshold', 'regime_threshold', 'atr']].notna().all(axis=1)
        active = ready & (out['atr'] > 0.0) & (out['weekly_corr'] > 0.0) & (out['weekly_corr'] > out['daily_corr'].abs()) & (out['weekly_corr'] > out['regime_threshold']) & (out['echo_magnitude'] > out['echo_threshold'])
        out['entry'] = np.select([active & (out['echo'] > 0.0) & (out['predicted_slope'] > 0.0), active & (out['echo'] < 0.0) & (out['predicted_slope'] < 0.0)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0):
                position = 0
                age = 0
                high_water = low_water = stop = np.nan
                continue
            if position:
                age += 1
                high_water = close if np.isnan(high_water) else max(high_water, close)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                if position == 1:
                    candidate = high_water - params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close <= stop
                else:
                    candidate = low_water + params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.max_hold:
                    position = 0
                    age = 0
                    high_water = low_water = stop = np.nan
                signals[i] = position
                continue
            if entries[i] in (-1, 1):
                position = int(entries[i])
                age = 0
                # Watermarks begin on the next bar, when the shifted entry fills.
                high_water = low_water = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EchoParams()
    index = pd.date_range('2020-01-01', periods=6)
    for direction, closes in [(1, [100, 100, 105, 104, 102, 100]), (-1, [100, 100, 95, 96, 98, 100])]:
        data = pd.DataFrame({'close': closes}, index=index)
        derived = pd.DataFrame({'atr': [1, 1, 1, 3, 3, 3], 'entry': [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        # Increasing ATR must not loosen the stop; exit decisions fill one bar later.
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0]
        assert (result['size'] > 0.0).all()
    params.max_hold = 2
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
