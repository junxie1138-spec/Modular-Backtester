from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    roc_window: int = 10
    acceleration_lag: int = 3
    atr_window: int = 14
    pressure_window: int = 10
    acceleration_threshold: float = 0.6
    retreat_ratio: float = 0.75
    trailing_k: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = 'gen_local_1791264395'

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(200, params.roc_window + params.acceleration_lag + 2,
                       params.atr_window + params.pressure_window + 3))

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if min(params.roc_window, params.acceleration_lag, params.atr_window,
               params.pressure_window, params.holding_bars) < 1:
            raise ValueError('Lookbacks and holding_bars must be positive')
        if not (np.isfinite(params.acceleration_threshold)
                and params.acceleration_threshold > 0
                and np.isfinite(params.retreat_ratio)
                and 0 < params.retreat_ratio < 1
                and np.isfinite(params.trailing_k) and params.trailing_k > 0):
            raise ValueError('Invalid acceleration, retreat, or trailing parameters')
        close = data['close']
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        safe_atr = out['atr'].where(out['atr'] > 0)
        out['roc'] = close.pct_change(params.roc_window, fill_method=None)
        out['acceleration'] = out['roc'].diff(params.acceleration_lag).div(
            safe_atr.div(close.where(close > 0)))
        out['prior_acceleration'] = out['acceleration'].shift(1)
        out['down_excursion'] = (data['open'] - data['low']).clip(lower=0).div(safe_atr)
        out['up_excursion'] = (data['high'] - data['open']).clip(lower=0).div(safe_atr)
        for side in ('down', 'up'):
            excursion = out[side + '_excursion']
            out[side + '_recent'] = excursion.rolling(3, min_periods=3).mean()
            out[side + '_baseline'] = excursion.shift(3).rolling(
                params.pressure_window, min_periods=params.pressure_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out.notna().all(axis=1) & (out['atr'] > 0)
        out['long_entry'] = (out['ready'] & (close > out['ma200'])
            & (out['roc'] > 0)
            & (out['acceleration'] > params.acceleration_threshold)
            & (out['acceleration'] > out['prior_acceleration'])
            & (out['down_recent'] < params.retreat_ratio * out['down_baseline']))
        out['short_entry'] = (out['ready'] & (close < out['ma200'])
            & (out['roc'] < 0)
            & (out['acceleration'] < -params.acceleration_threshold)
            & (out['acceleration'] < out['prior_acceleration'])
            & (out['up_recent'] < params.retreat_ratio * out['up_baseline']))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        regimes = indicators['ma200'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        for i, close in enumerate(closes):
            if not ready[i] or not np.isfinite(close) or close <= 0:
                position = 0
                age = 0
                continue
            if position:
                # The first held close follows the entry decision by one bar.
                if age == 0:
                    extreme = close
                    stop = close - position * params.trailing_k * atrs[i]
                age += 1
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trailing_k * atrs[i])
                    stopped = close <= stop
                    regime_exit = close <= regimes[i]
                else:
                    # Mirror the high-close trail with a low-close trail for shorts.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trailing_k * atrs[i])
                    stopped = close >= stop
                    regime_exit = close >= regimes[i]
                if stopped or regime_exit or age >= params.holding_bars:
                    position = 0
                    age = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Verify delayed entry and a trail that cannot loosen when ATR expands.
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [101., 104., 103., 105., 101., 101.]}, index=index)
    derived = pd.DataFrame({
        'atr': [1., 1., 1., 1., 10., 1.], 'ma200': 100., 'ready': True,
        'long_entry': [True, False, False, False, False, False],
        'short_entry': False
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, PressureParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
