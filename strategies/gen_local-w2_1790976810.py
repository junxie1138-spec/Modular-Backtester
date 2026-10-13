from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterDrawdownParams:
    peak_window: int = 42
    atr_window: int = 14
    min_depth_atr: float = 1.5
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.peak_window < 2 or self.atr_window < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if self.hold_bars < 1 or self.min_depth_atr <= 0 or self.stop_atr <= 0:
            raise ValueError('Holding period, depth threshold and stop must be positive.')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[QuarterDrawdownParams]):
    strategy_id = 'gen_local-w2_1790976810'

    @classmethod
    def params_type(cls):
        return QuarterDrawdownParams

    def warmup_bars(self, params: QuarterDrawdownParams) -> int:
        return int(max(params.peak_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: QuarterDrawdownParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['peak'] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out['drawdown'] = out['peak'] - close
        out['drawdown_change'] = out['drawdown'].diff()
        out['depth_atr'] = out['drawdown'] / out['atr'].where(out['atr'] > 0)
        month = pd.Series(data.index.month, index=data.index)
        day = pd.Series(data.index.day, index=data.index)
        out['quarter_end_phase'] = ((month % 3 == 0) & (day >= 15)).astype(int)
        out['quarter_start_phase'] = ((month % 3 == 1) & (day <= 12)).astype(int)
        valid = np.isfinite(out[['atr', 'depth_atr', 'drawdown_change', 'previous_close']]).all(axis=1)
        eligible = valid & (out['atr'] > 0) & (out['depth_atr'] >= params.min_depth_atr)
        recovery = (out['drawdown_change'] < 0) & (close > out['previous_close'])
        deterioration = (out['drawdown_change'] > 0) & (close < out['previous_close'])
        out['entry_direction'] = np.select([
            eligible & out['quarter_end_phase'].eq(1) & recovery,
            eligible & out['quarter_start_phase'].eq(1) & deterioration,
        ], [1, -1], default=0).astype(int)
        out['strength'] = (out['depth_atr'] / (3.0 * params.min_depth_atr)).clip(0, 1).fillna(0)
        out['entry_size'] = params.min_size + (params.max_size - params.min_size) * out['strength']
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterDrawdownParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        directions = indicators['entry_direction'].to_numpy(dtype=int)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan
        fixed_stop = np.nan
        position_size = params.min_size
        for i in range(count):
            exited = False
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                    fixed_stop = entry_price - side * stop_distance
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                stopped = (side == 1 and closes[i] < fixed_stop) or (side == -1 and closes[i] > fixed_stop)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    side = 0
                    exited = True
            if not side and not exited and directions[i] != 0:
                if np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(entry_sizes[i]):
                    side = int(directions[i])
                    entry_bar = i + 1
                    entry_price = np.nan
                    stop_distance = params.stop_atr * atrs[i]
                    position_size = float(np.clip(entry_sizes[i], params.min_size, params.max_size))
            signals[i] = side
            if side:
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-03-16', periods=8, freq='B')
    data = pd.DataFrame({
        'open': [100.0] * 8,
        'high': [104.0] * 8,
        'low': [96.0] * 8,
        'close': [100.0, 100.0, 97.0, 100.0, 100.0, 100.0, 103.0, 100.0],
        'volume': [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        'entry_direction': [1, 0, 0, 0, -1, 0, 0, 0],
        'atr': [1.0] * 8,
        'entry_size': [0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8],
    }, index=index)
    strategy = GeneratedStrategy()
    params = QuarterDrawdownParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert np.allclose(result['size'], [0.25, 0.5, 0.5, 0.25, 0.25, 0.8, 0.8, 0.25])
    expiry = strategy.generate_signals(data, indicators, ctx, QuarterDrawdownParams(hold_bars=1)).data
    assert expiry['signal'].tolist() == [0, 1, 0, 0, 0, -1, 0, 0]
    computed = strategy.indicators(data, params)
    assert computed['entry_direction'].eq(0).all()
    assert np.isfinite(computed['entry_size']).all()


if __name__ == '__main__':
    _self_check()
