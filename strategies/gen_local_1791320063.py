from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 4
    vol_bars: int = 20
    atr_bars: int = 14
    area_threshold: float = 0.10
    target_vol: float = 0.10
    trail_atr: float = 1.5
    hold_bars: int = 2

    def __post_init__(self):
        if min(self.path_bars, self.vol_bars, self.atr_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if self.hold_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions')
        if not 0.0 < self.area_threshold < 0.5:
            raise ValueError('Area threshold must be between zero and 0.5')
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError('Target volatility must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('ATR multiplier must be finite and positive')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791320063'

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.path_bars + 1, params.vol_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data, params):
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        gap = np.log(opening / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        intraday = np.log(close / opening).replace([np.inf, -np.inf], np.nan)
        length = params.path_bars
        gap_path = gap.fillna(0.0).cumsum()
        day_path = intraday.fillna(0.0).cumsum()
        gap_sum = gap.rolling(length, min_periods=length).sum()
        day_sum = intraday.rolling(length, min_periods=length).sum()
        cross = gap_path.shift(1).fillna(0.0) * intraday - day_path.shift(1).fillna(0.0) * gap
        area = 0.5 * (cross.rolling(length, min_periods=length).sum()
                      - gap_path.shift(length).fillna(0.0) * day_sum
                      + day_path.shift(length).fillna(0.0) * gap_sum)
        scale = gap.abs().rolling(length, min_periods=length).sum() * intraday.abs().rolling(length, min_periods=length).sum()
        previous = close.shift(1)
        true_range = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        returns = close.pct_change(fill_method=None)
        return pd.DataFrame({
            'gap': gap,
            'intraday': intraday,
            'gap_sum': gap_sum,
            'intraday_sum': day_sum,
            'signature_area': area / scale.where(scale > 0),
            'atr': true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean(),
            'volatility': returns.rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0) * np.sqrt(252.0),
            'ma200': close.rolling(200, min_periods=200).mean()
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1)
        regime = (data['close'] > indicators['ma200']).to_numpy(dtype=bool)
        entry = (ready & regime
                 & (indicators['signature_area'] < -params.area_threshold).to_numpy()
                 & (indicators['gap_sum'] < 0).to_numpy()
                 & (indicators['intraday_sum'] > 0).to_numpy()
                 & (indicators['intraday'] > 0).to_numpy()
                 & (indicators['atr'] > 0).to_numpy()
                 & (indicators['volatility'] > 0).to_numpy())
        exposure = (params.target_vol / indicators['volatility'].where(indicators['volatility'] > 0)).clip(0.01, 0.95).fillna(0.95).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        trade_size = 0.95
        for i in range(len(data)):
            if active:
                if not ready[i] or not np.isfinite(close[i]) or not regime[i]:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or i - entry_bar >= params.hold_bars:
                        active = False
            elif entry[i]:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                trade_size = float(exposure[i])
            signals[i] = int(active)
            if active:
                sizes[i] = trade_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 102.0, 98.0, 100.0, 101.0, 102.0]}, index=index)
    derived = pd.DataFrame({
        'signature_area': [-0.3, 0.0, 0.0, -0.3, 0.0, 0.0],
        'gap_sum': -0.02, 'intraday_sum': 0.03, 'intraday': 0.01,
        'atr': 1.0, 'volatility': 0.2, 'ma200': 90.0
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, SignatureParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    assert np.isclose(result['size'].iloc[1], 0.5)
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
