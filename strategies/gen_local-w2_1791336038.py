from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    roc_bars: int = 2
    noise_bars: int = 20
    shock_z: float = 1.75
    atr_bars: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = 'gen_local-w2_1791336038'

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return int(max(200, params.roc_bars + params.noise_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        if params.roc_bars < 1 or params.noise_bars < 2 or params.atr_bars < 1:
            raise ValueError('Lookbacks must be positive; noise_bars must be at least two.')
        if not np.isfinite(params.shock_z) or params.shock_z <= 0:
            raise ValueError('shock_z must be finite and positive.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.hold_bars not in (1, 2):
            raise ValueError('stop_atr must be finite and positive; hold_bars must be one or two.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous_close = close.shift(1)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['roc'] = close.div(close.shift(params.roc_bars).replace(0.0, np.nan)).sub(1.0)
        out['previous_roc'] = out['roc'].shift(1)
        out['acceleration'] = out['roc'].diff()
        out['prior_noise'] = out['acceleration'].pow(2).rolling(params.noise_bars, min_periods=params.noise_bars).mean().shift(1).pow(0.5)
        out['acceleration_z'] = out['acceleration'].div(out['prior_noise'].replace(0.0, np.nan))
        ranges = pd.concat([
            data['high'].sub(data['low']),
            data['high'].sub(previous_close).abs(),
            data['low'].sub(previous_close).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['entry'] = (
            close.gt(out['ma200'])
            & out['roc'].lt(0.0)
            & out['previous_roc'].ge(0.0)
            & out['acceleration_z'].le(-params.shock_z)
            & out['atr'].gt(0.0)
            & np.isfinite(out['atr'])
            & np.isfinite(out['acceleration_z'])
            & np.isfinite(close)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        if params.hold_bars not in (1, 2) or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Invalid holding period or stop multiplier.')
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=np.int64)
        active = False
        age = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if age == 0:
                    # The preceding close requested entry; this bar's open is the fill reference.
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]):
                        active = False
                        continue
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or age >= params.hold_bars:
                    active = False
                    age = 0
                    continue
                desired[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                age = 0
                desired[i] = 1
        df = pd.DataFrame({'signal': desired, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShockParams()
    index = pd.date_range('2020-01-01', periods=5, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 95.0, 'close': [100.0, 99.0, 99.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False], 'atr': [2.0, 0.1, 0.1, 0.1, 0.1]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], 'close'] = 97.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= 200
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
