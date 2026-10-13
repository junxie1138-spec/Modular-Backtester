from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    loss_window: int = 5
    context_window: int = 20
    atr_window: int = 14
    concentration_min: float = 0.65
    shock_sigma: float = 1.5
    min_drawdown: float = 0.01
    trail_atr: float = 1.5
    hold_days: int = 4

    def __post_init__(self):
        windows = (self.loss_window, self.context_window, self.atr_window)
        if any(not isinstance(value, int) or value < 2 for value in windows):
            raise ValueError('Lookback windows must be integers >= 2.')
        if not isinstance(self.hold_days, int) or not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be an integer from 3 to 5.')
        if not np.isfinite(self.concentration_min) or not 0 < self.concentration_min <= 1:
            raise ValueError('concentration_min must lie in (0, 1].')
        if not np.isfinite(self.min_drawdown) or not 0 < self.min_drawdown < 1:
            raise ValueError('min_drawdown must lie in (0, 1).')
        if any(not np.isfinite(value) or value <= 0 for value in (self.shock_sigma, self.trail_atr)):
            raise ValueError('Shock and trailing-stop multipliers must be positive and finite.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791244431'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.loss_window + 2, params.context_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['loss'] = -out['return'].clip(upper=0)
        prior_loss = out['loss'].shift(1)
        out['loss_mass'] = prior_loss.rolling(params.loss_window, min_periods=params.loss_window).sum()
        out['shock'] = prior_loss.rolling(params.loss_window, min_periods=params.loss_window).max()
        prior_return = out['return'].shift(1)
        out['prior_net_return'] = prior_return.rolling(params.loss_window, min_periods=params.loss_window).sum()
        out['prior_volatility'] = prior_return.rolling(params.context_window, min_periods=params.context_window).std(ddof=0)
        out['concentration'] = out['shock'].div(out['loss_mass'].where(out['loss_mass'] > 0))
        out['repair_fraction'] = out['return'].div(out['shock'].where(out['shock'] > 0))
        out['rolling_high'] = close.rolling(params.context_window, min_periods=params.context_window).max()
        out['drawdown'] = 1.0 - close.div(out['rolling_high'])
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['conviction'] = (
            out['concentration'].clip(0.0, 1.0)
            * (out['repair_fraction'] / 0.65).clip(0.0, 1.0)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['entry'] = (
            (out['prior_net_return'] < 0)
            & (out['drawdown'] >= params.min_drawdown)
            & (out['concentration'] >= params.concentration_min)
            & (out['prior_volatility'] > 0)
            & (out['shock'] >= params.shock_sigma * out['prior_volatility'])
            & out['repair_fraction'].between(0.15, 0.65)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        active = False
        highest = stop = -np.inf
        age = 0
        position_size = 0.25
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                active = False
                continue
            if active:
                age += 1
                highest = max(highest, close)
                stop = max(stop, highest - params.trail_atr * atr)
                if close <= stop or age >= params.hold_days:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i]:
                active = True
                age = 0
                # The position fills next bar; its closing high starts there.
                highest = stop = -np.inf
                position_size = 0.25 + 0.70 * conviction[i]
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    close = pd.Series([100., 101., 104., 103., 101., 100., 99., 100.], index=index)
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.})
    indicators = pd.DataFrame({
        'atr': [1., 1., 1., 10., 10., 10., 10., 10.],
        'entry': [True, False, False, False, False, False, False, False],
        'conviction': [0.8] * 8,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=22)
    result = strategy.generate_signals(data, indicators, ctx, RecoveryParams(hold_days=5)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.allclose(result.loc[index[1:5], 'size'], 0.81)
    result = strategy.generate_signals(data, indicators, ctx, RecoveryParams(hold_days=3)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    warmup = strategy.indicators(data, RecoveryParams())
    assert not warmup['entry'].any()
    assert warmup['conviction'].notna().all()
    assert strategy.warmup_bars(RecoveryParams()) == 22


if __name__ == '__main__':
    _self_check()
