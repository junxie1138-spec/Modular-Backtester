from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilitySwitchParams:
    fast_window: int = 8
    slow_window: int = 40
    atr_window: int = 14
    quiet_ratio: float = 0.85
    elevated_ratio: float = 1.10
    dominance_threshold: float = 0.25
    intraday_atr_threshold: float = 0.15
    spike_atr: float = 2.50
    refractory_bars: int = 3
    trailing_atr: float = 1.80
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VolatilitySwitchParams]):
    strategy_id = 'gen_local_1791335550'

    @classmethod
    def params_type(cls) -> type[VolatilitySwitchParams]:
        return VolatilitySwitchParams

    def warmup_bars(self, params: VolatilitySwitchParams) -> int:
        return int(max(params.fast_window, params.slow_window, params.atr_window) + params.refractory_bars + 2)

    def indicators(self, data: pd.DataFrame, params: VolatilitySwitchParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous = close.shift(1)
        out['return'] = close.div(previous.where(previous > 0)).sub(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out['fast_vol'] = out['return'].rolling(p.fast_window, min_periods=p.fast_window).std(ddof=0)
        out['slow_vol'] = out['return'].rolling(p.slow_window, min_periods=p.slow_window).std(ddof=0)
        out['vol_ratio'] = out['fast_vol'].div(out['slow_vol'].where(out['slow_vol'] > 0))
        out['vol_change'] = out['vol_ratio'].diff()
        out['up_variance'] = out['return'].clip(lower=0).pow(2).rolling(p.fast_window, min_periods=p.fast_window).mean()
        out['down_variance'] = out['return'].clip(upper=0).pow(2).rolling(p.fast_window, min_periods=p.fast_window).mean()
        total = out['up_variance'] + out['down_variance']
        out['dominance'] = (out['up_variance'] - out['down_variance']).div(total.where(total > 0))
        out['intraday_move'] = (close - data['open']).div(out['atr'].where(out['atr'] > 0))
        out['spike_multiple'] = out['true_range'].div(out['atr'].shift(1).where(out['atr'].shift(1) > 0))
        out['recent_spike'] = out['spike_multiple'].ge(p.spike_atr).astype(float).rolling(p.refractory_bars, min_periods=p.refractory_bars).max()
        out['quiet'] = out['vol_ratio'].le(p.quiet_ratio)
        out['cooling'] = out['vol_ratio'].ge(p.elevated_ratio) & out['vol_change'].lt(0)
        volatility_direction = pd.Series(np.select([
            out['dominance'].gt(p.dominance_threshold),
            out['dominance'].lt(-p.dominance_threshold),
        ], [1, -1], default=0), index=data.index)
        out['volatility_direction'] = np.select([
            out['quiet'], out['cooling'],
        ], [volatility_direction, -volatility_direction], default=0)
        out['intraday_direction'] = np.select([
            out['intraday_move'].gt(p.intraday_atr_threshold),
            out['intraday_move'].lt(-p.intraday_atr_threshold),
        ], [1, -1], default=0)
        required = ['atr', 'vol_ratio', 'vol_change', 'dominance', 'intraday_move', 'spike_multiple', 'recent_spike']
        out['ready'] = np.isfinite(out[required]).all(axis=1) & out['atr'].gt(0)
        agreement = out['volatility_direction'].eq(out['intraday_direction'])
        out['candidate'] = out['volatility_direction'].where(
            out['ready'] & agreement & out['recent_spike'].eq(0), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilitySwitchParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        position = 0
        age = 0
        high_water = low_water = stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not ready[i] or not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trailing_atr * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.trailing_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                age = 0
                high_water = low_water = close
                stop = close - position * params.trailing_atr * atr
            signals[i] = position
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VolatilitySwitchParams()
    index = pd.date_range('2020-01-01', periods=9, freq='D')
    close = pd.Series([100., 101., 104., 101., 100., 99., 98., 100., 101.], index=index)
    data = pd.DataFrame({'open': close, 'high': close + 0.5, 'low': close - 0.5, 'close': close, 'volume': 1000.}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'ready': True, 'candidate': [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    indicators['candidate'] = 0
    indicators.loc[index[0], 'candidate'] = 1
    indicators['atr'] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    flat = pd.concat([data.iloc[[0]]] * 80, ignore_index=True)
    flat.index = pd.date_range('2020-01-01', periods=80, freq='D')
    computed = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, computed, None, params).data
    assert result.index.equals(flat.index)
    assert result['signal'].eq(0).all()
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
