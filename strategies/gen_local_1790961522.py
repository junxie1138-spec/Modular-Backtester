from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    history: int = 252
    holding_bars: int = 8
    min_matches: int = 8
    min_mean_return: float = 0.003
    min_win_rate: float = 0.56
    atr_bars: int = 14
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1790961522'

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(200, 11, params.atr_bars + 1) + params.holding_bars + 1 + params.history)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if params.history < 1 or params.holding_bars < 1 or params.atr_bars < 1:
            raise ValueError('Lookbacks and holding_bars must be positive.')
        if params.min_matches < 1 or params.min_matches > params.history:
            raise ValueError('min_matches must be between 1 and history.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive.')
        if not np.isfinite(params.min_mean_return) or not 0 <= params.min_win_rate <= 1:
            raise ValueError('Invalid analog thresholds.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['regime'] = close.gt(out['ma200']) & np.isfinite(out['ma200'])
        previous = close.shift(1)
        out['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['tr'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        for horizon in (2, 5, 10):
            out[f'roc{horizon}'] = close.pct_change(horizon, fill_method=None)
            out[f'acc{horizon}'] = out[f'roc{horizon}'].diff()
        out['contracting'] = out['tr'].lt(out['atr'])
        shape_valid = np.isfinite(out[['acc2', 'acc5', 'acc10', 'roc10', 'atr']]).all(axis=1)
        out['phase'] = (
            out['acc2'].gt(0).astype(int)
            + 2 * out['acc5'].gt(0).astype(int)
            + 4 * out['acc10'].gt(0).astype(int)
            + 8 * out['roc10'].ge(0).astype(int)
            + 16 * out['contracting'].astype(int)
        ).astype(float).where(shape_valid)
        delay = params.holding_bars + 1
        out['mature_phase'] = out['phase'].shift(delay)
        out['mature_regime'] = out['regime'].shift(delay).eq(True)
        # A decision at s enters at open[s+1] and exits at open[s+holding_bars+1].
        out['mature_return'] = data['open'].div(data['open'].shift(params.holding_bars)).sub(1)
        out['mature_return'] = out['mature_return'].replace([np.inf, -np.inf], np.nan)
        out['analog_count'] = np.nan
        out['analog_mean'] = np.nan
        out['analog_win_rate'] = np.nan
        # Phase 3: short and medium acceleration positive, long acceleration and ROC negative.
        for phase in (3, 19):
            eligible = (
                out['mature_phase'].eq(phase)
                & out['mature_regime']
                & np.isfinite(out['mature_return'])
            )
            out[f'match_{phase}'] = eligible
            count = eligible.astype(float).rolling(params.history, min_periods=params.history).sum()
            total = out['mature_return'].where(eligible, 0.0).rolling(params.history, min_periods=params.history).sum()
            wins = (eligible & out['mature_return'].gt(0)).astype(float).rolling(params.history, min_periods=params.history).sum()
            denominator = count.replace(0, np.nan)
            out[f'count_{phase}'] = count
            out[f'mean_{phase}'] = total.div(denominator)
            out[f'win_rate_{phase}'] = wins.div(denominator)
            current = out['phase'].eq(phase)
            out.loc[current, 'analog_count'] = count.loc[current]
            out.loc[current, 'analog_mean'] = out.loc[current, f'mean_{phase}']
            out.loc[current, 'analog_win_rate'] = out.loc[current, f'win_rate_{phase}']
        out['ready'] = np.arange(len(data)) >= self.warmup_bars(params) - 1
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        entry = (
            indicators['ready']
            & indicators['regime']
            & indicators['phase'].isin([3, 19])
            & indicators['analog_count'].ge(params.min_matches)
            & indicators['analog_mean'].ge(params.min_mean_return)
            & indicators['analog_win_rate'].ge(params.min_win_rate)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        regimes = indicators['regime'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # The first held close occurs after the entry decision's next-open fill.
                    high_water = price if np.isnan(high_water) else max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if price <= stop or age >= params.holding_bars or not regimes[i]:
                        active = False
                else:
                    active = False
            elif valid and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ShapeParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100., 100., 103., 102., 100., 100.],
        'high': [101., 101., 104., 103., 101., 101.],
        'low': [99., 99., 102., 101., 99., 99.],
        'close': [100., 100., 103., 102., 100., 100.],
        'volume': 1000.,
    }, index=index)
    indicators = pd.DataFrame({
        'ready': True, 'regime': True,
        'phase': [3., 0., 0., 0., 0., 0.],
        'analog_count': 20., 'analog_mean': 0.02,
        'analog_win_rate': 0.8,
        'atr': [1., 1., 1., 5., 5., 5.],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    params.holding_bars = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
