from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnCentroidParams:
    lookback: int = 12
    drawdown_min: float = 0.02
    centroid_gap: float = 0.25
    atr_period: int = 14
    atr_mult: float = 1.1
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[ReturnCentroidParams]):
    strategy_id = 'gen_local-w2_1791246313'

    @classmethod
    def params_type(cls):
        return ReturnCentroidParams

    def warmup_bars(self, params: ReturnCentroidParams) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnCentroidParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1 or params.max_hold < 1:
            raise ValueError('Lookback must be >= 2; ATR period and holding limit must be positive.')
        if not 0.0 < params.drawdown_min < 1.0 or not 0.0 <= params.centroid_gap <= 1.0:
            raise ValueError('Invalid drawdown or centroid threshold.')
        if not np.isfinite(params.atr_mult) or params.atr_mult <= 0.0:
            raise ValueError('ATR multiplier must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['return'] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out['gain'] = out['return'].clip(lower=0.0)
        out['loss'] = -out['return'].clip(upper=0.0)
        position = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out['gain_weighted'] = out['gain'] * position
        out['loss_weighted'] = out['loss'] * position
        length = params.lookback
        out['gain_mass'] = out['gain'].rolling(length, min_periods=length).sum()
        out['loss_mass'] = out['loss'].rolling(length, min_periods=length).sum()
        out['gain_moment'] = out['gain_weighted'].rolling(length, min_periods=length).sum()
        out['loss_moment'] = out['loss_weighted'].rolling(length, min_periods=length).sum()
        origin = position - length + 1.0
        out['gain_centroid'] = (out['gain_moment'] / out['gain_mass'].where(out['gain_mass'] > 0.0) - origin) / (length - 1.0)
        out['loss_centroid'] = (out['loss_moment'] / out['loss_mass'].where(out['loss_mass'] > 0.0) - origin) / (length - 1.0)
        out['centroid_gap'] = out['gain_centroid'] - out['loss_centroid']
        out['window_return'] = out['return'].rolling(length, min_periods=length).sum()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        valid = np.isfinite(out[['return', 'centroid_gap', 'window_return', 'atr']]).all(axis=1)
        out['setup'] = (
            valid
            & out['return'].gt(0.0)
            & out['window_return'].le(np.log1p(-params.drawdown_min))
            & out['centroid_gap'].ge(params.centroid_gap)
            & out['atr'].gt(0.0)
        )
        out['confirmed'] = out['setup'] & out['setup'].shift(1, fill_value=False)
        out['entry'] = out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnCentroidParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        held = 0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                held += 1
                if not valid:
                    active = False
                else:
                    # The first active bar is the fill bar; exclude the pre-entry decision close.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_mult * atr)
                    if close <= stop or held >= params.max_hold:
                        active = False
                signals[i] = int(active)
            elif valid and entries[i]:
                active = True
                high_water = -np.inf
                stop = -np.inf
                held = 0
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 102.0, 100.0, 99.0, 98.0, 97.0]}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0, 10.0, 10.0, 10.0, 10.0], 'entry': [True, False, False, False, False, False]}, index=index)
    # Wider ATR must not loosen the stop established at the first in-trade close.
    result = strategy.generate_signals(data, indicators, None, ReturnCentroidParams(max_hold=5))
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    indicators['atr'] = 100.0
    result = strategy.generate_signals(data, indicators, None, ReturnCentroidParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert result.data['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
