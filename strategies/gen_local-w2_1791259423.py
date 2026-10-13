from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    min_links: int = 3
    persistence: float = 0.35
    long_share: float = 0.35
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1791259423'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.graph_window + 2, params.atr_window + 1))

    @staticmethod
    def indicators(data, params):
        if (params.graph_window < 3 or params.atr_window < 1
                or params.min_links < 1 or params.hold_bars < 1
                or not 0 <= params.persistence <= 1
                or not 0 <= params.long_share <= 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['shock'] = out['return'].abs()
        direction = np.sign(out['return'])
        degree = pd.Series(0.0, index=data.index)
        long_degree = degree.copy()
        agreement = degree.copy()
        for lag in range(1, params.graph_window + 1):
            past = out['shock'].shift(lag)
            if lag == 1:
                visible = out['shock'].notna() & past.notna()
            else:
                blocker = out['shock'].shift(1).rolling(lag - 1, min_periods=lag - 1).max()
                visible = blocker.lt(out['shock']) & blocker.lt(past)
            out[f'edge_{lag}'] = visible.astype(float)
            degree += visible.astype(float)
            agreement += (visible.astype(float) * direction * direction.shift(lag)).fillna(0.0)
            if lag >= 3:
                long_degree += visible.astype(float)
        out['degree'] = degree
        out['long_degree'] = long_degree
        out['persistence'] = (agreement / degree.replace(0, np.nan)).fillna(0.0)
        out['long_share'] = (long_degree / degree.replace(0, np.nan)).fillna(0.0)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = (out['return'].shift(params.graph_window).notna()
                 & out['atr'].gt(0) & np.isfinite(out['atr']))
        eligible = (ready & degree.ge(params.min_links)
                    & out['persistence'].ge(params.persistence)
                    & out['long_share'].ge(params.long_share))
        out['candidate'] = direction.where(eligible, 0).fillna(0).astype(int)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        atr = indicators['atr'].to_numpy(dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        entry_bar = -1
        stop = -np.inf
        watermark = -np.inf
        armed = False
        for i, (opening, high, low, close) in enumerate(prices):
            exited = False
            valid = np.isfinite([opening, high, low, close, atr[i]]).all() and opening > 0 and atr[i] > 0
            if position:
                if np.isnan(entry):
                    # The preceding close requested entry; this bar supplies its actual open.
                    if i == 0 or not valid or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        position = 0
                        exited = True
                    else:
                        entry = opening
                        entry_bar = i
                        watermark = position * entry
                        stop = watermark - params.trail_atr * atr[i - 1]
                        armed = False
                if position:
                    if not valid:
                        position = 0
                        exited = True
                    else:
                        # Signed price makes the protective stop increase for either direction.
                        adverse = position * (low if position == 1 else high)
                        favorable = position * (high if position == 1 else low)
                        signed_entry = position * entry
                        if adverse <= stop:
                            position = 0
                            exited = True
                        else:
                            watermark = max(watermark, favorable)
                            if watermark - signed_entry >= entry * params.breakeven_pct:
                                armed = True
                            if armed:
                                stop = max(stop, signed_entry, watermark - params.trail_atr * atr[i])
                            # Newly raised stops are checked at the close, without assuming intrabar order.
                            if position * close <= stop or i - entry_bar + 1 >= params.hold_bars:
                                position = 0
                                exited = True
            if not position and not exited and valid and confirmed[i] != 0:
                position = int(confirmed[i])
                entry = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [101.0, 101.0, 101.0, 104.0, 104.0, 101.0, 101.0],
        'low': [99.0] * 7,
        'close': [100.0, 100.0, 100.0, 103.0, 102.0, 100.0, 100.0],
        'volume': [1000.0] * 7
    }, index=index)
    ind = pd.DataFrame({'candidate': [1, 1, 0, 0, 0, 0, 0], 'atr': 1.0}, index=index)
    params = VisibilityParams()
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0]
    mirrored = data.copy()
    mirrored['open'] = 200 - data['open']
    mirrored['high'] = 200 - data['low']
    mirrored['low'] = 200 - data['high']
    mirrored['close'] = 200 - data['close']
    ind['candidate'] *= -1
    short = GeneratedStrategy.generate_signals(mirrored, ind, None, params).data
    assert short['signal'].tolist() == [0, 0, -1, -1, -1, 0, 0]
    assert result['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.graph_window + 2


if __name__ == '__main__':
    _self_check()
