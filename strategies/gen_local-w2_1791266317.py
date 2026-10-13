from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    ma_window: int = 12
    correlation_window: int = 30
    z_threshold: float = 1.3
    correlation_threshold: float = 0.1
    atr_window: int = 14
    stop_atr: float = 1.5
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = 'gen_local-w2_1791266317'

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(200, params.ma_window + params.correlation_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if min(params.ma_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if params.holding_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions.')
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError('z_threshold must be finite and positive.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        if not np.isfinite(params.correlation_threshold) or not 0 <= params.correlation_threshold < 1:
            raise ValueError('correlation_threshold must be in [0, 1).')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['std'].replace(0.0, np.nan)
        out['displacement_change'] = out['z'].diff()
        out['lag_one'] = out['displacement_change'].shift(1)
        out['lag_two'] = out['displacement_change'].shift(2)
        out['ac_one'] = out['displacement_change'].rolling(params.correlation_window).corr(out['lag_one'])
        out['ac_two'] = out['displacement_change'].rolling(params.correlation_window).corr(out['lag_two'])
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['regime_ma'] = close.rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        required = indicators[['z', 'ac_one', 'ac_two', 'atr', 'regime_ma']]
        ready = pd.Series(np.isfinite(required.to_numpy(dtype=float)).all(axis=1), index=data.index)
        oscillating = ready & (indicators['atr'] > 0) & (indicators['ac_one'] < -params.correlation_threshold) & (indicators['ac_two'] > params.correlation_threshold)
        long_entry = oscillating & (indicators['z'] < -params.z_threshold) & (data['close'] > indicators['regime_ma'])
        short_entry = oscillating & (indicators['z'] > params.z_threshold) & (data['close'] < indicators['regime_ma'])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan
        for i in range(len(data)):
            if position:
                # Observe the execution open only when its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or not np.isfinite(close[i])
                # Both directions retain the entry decision's ATR permanently.
                stopped = position * (close[i] - entry_price) < -stop_distance
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid or stopped or expired:
                    position = 0
                raw[i] = position
            elif candidates[i] and np.isfinite(close[i]):
                position = int(candidates[i])
                entry_bar = i + 1
                stop_distance = params.stop_atr * atr[i]
                raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[1], 'close'] = 97.0
    data.loc[index[4], 'close'] = 103.0
    ind = pd.DataFrame({'z': 0.0, 'ac_one': -0.5, 'ac_two': 0.5, 'atr': 1.0, 'regime_ma': 99.0}, index=index)
    ind.loc[index[[0, 6]], 'z'] = -2.0
    ind.loc[index[3], ['z', 'regime_ma']] = [2.0, 101.0]
    result = GeneratedStrategy().generate_signals(data, ind, None, ShockwaveParams()).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
