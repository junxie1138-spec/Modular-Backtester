from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 20
    atr_window: int = 14
    pressure_window: int = 4
    depth_atr: float = 2.0
    overnight_share: float = 0.65
    trail_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local-w2_1790941325'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.peak_window, params.atr_window, params.pressure_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.peak_window, params.atr_window, params.pressure_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not (params.depth_atr > 0 and params.trail_atr > 0):
            raise ValueError('ATR thresholds must be positive.')
        if not 0 < params.overnight_share <= 1 or params.max_hold not in (1, 2):
            raise ValueError('Invalid concentration or holding horizon.')

        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['previous_close'] = previous
        out['peak'] = previous.rolling(params.peak_window).max()
        out['gap'] = data['open'] - previous
        out['body'] = close - data['open']
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean().replace(0.0, np.nan)
        out['depth'] = ((out['peak'] - close) / out['atr']).clip(lower=0.0)
        out['overnight_loss'] = (-out['gap']).clip(lower=0.0)
        out['intraday_loss'] = (-out['body']).clip(lower=0.0)
        out['overnight_budget'] = out['overnight_loss'].rolling(params.pressure_window).sum()
        out['intraday_budget'] = out['intraday_loss'].rolling(params.pressure_window).sum()
        out['loss_budget'] = out['overnight_budget'] + out['intraday_budget']
        out['gap_share'] = out['overnight_budget'] / out['loss_budget'].replace(0.0, np.nan)
        out['recovery'] = (out['body'] / out['overnight_loss'].replace(0.0, np.nan)).clip(0.0, 1.0)
        out['strength'] = (
            (1.0 - np.exp(-out['depth'] / params.depth_atr))
            * out['gap_share'] * out['recovery']
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        out['entry'] = (
            (out['depth'] >= params.depth_atr)
            & (out['gap_share'] >= params.overnight_share)
            & (out['gap'] < 0.0)
            & (out['body'] > 0.0)
            & (close < previous)
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.2, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.0).to_numpy(dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        position_size = 0.2

        for i in range(count):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and closes[i] > 0.0 and atrs[i] > 0.0
            if active:
                age += 1
                if valid:
                    # Only closes after the pending entry count toward the trade high.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                if not valid or closes[i] <= stop or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
                # An exit cannot immediately reopen on the same decision bar.
                continue

            if valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                position_size = 0.2 + 0.75 * float(np.clip(strengths[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    close = np.array([105.0, 100.0, 99.0, 105.0, 110.0, 111.0])
    data = pd.DataFrame({
        'open': close, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'atr': 1.0,
        'entry': [True, False, False, True, False, False],
        'strength': [0.2, 0.0, 0.0, 0.8, 0.0, 0.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    assert np.isclose(result['size'].iloc[1], 0.35)
    assert np.isclose(result['size'].iloc[4], 0.8)
    assert result['size'].gt(0.0).all()
    prefix = strategy.generate_signals(data.iloc[:5], derived.iloc[:5], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:5])
    params.max_hold = 1
    one_day = strategy.generate_signals(data, derived, ctx, params).data
    assert one_day['signal'].tolist() == [0, 1, 0, 0, 1, 0]
    actual = strategy.indicators(data, GeneratedParams())
    warmup = strategy.generate_signals(data, actual, ctx, GeneratedParams()).data
    assert warmup['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
