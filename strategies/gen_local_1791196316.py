from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_bars: int = 20
    regime_bars: int = 20
    pressure_bars: int = 3
    efficiency_threshold: float = 0.35
    z_threshold: float = 0.8
    pressure_threshold: float = 0.15
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791196316'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_bars, params.regime_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out['price_std'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].where(out['price_std'] > 0)
        out['step'] = close.diff()
        out['path_length'] = out['step'].abs().rolling(params.regime_bars, min_periods=params.regime_bars).sum()
        out['net_displacement'] = close.diff(params.regime_bars)
        out['efficiency'] = out['net_displacement'] / out['path_length'].where(out['path_length'] > 0)
        out['range'] = data['high'] - data['low']
        out['body'] = close - data['open']
        out['body_pressure'] = out['body'] / out['range'].where(out['range'] > 0)
        out['pressure'] = out['body_pressure'].rolling(params.pressure_bars, min_periods=params.pressure_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['trend_regime'] = out['efficiency'] > params.efficiency_threshold
        out['congestion_regime'] = out['efficiency'].abs() <= params.efficiency_threshold
        out['displacement_agrees'] = (
            (out['trend_regime'] & (out['z'] >= params.z_threshold))
            | (out['congestion_regime'] & (out['z'] <= -params.z_threshold))
        )
        out['pressure_agrees'] = out['pressure'] >= params.pressure_threshold
        out['entry'] = (out['displacement_agrees'] & out['pressure_agrees']).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        eligible = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held += 1
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                if invalid_entry or target_hit or held >= params.max_hold:
                    active = False
                    entry_price = np.nan
                    held = 0
                else:
                    signal[i] = 1
            elif eligible[i]:
                active = True
                entry_price = np.nan
                held = 0
                signal[i] = 1
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(max_hold=5)
    index = pd.date_range('2020-01-01', periods=14, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True] + [False] * 13}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1] + [0] * 8
    data.loc[index[2], 'close'] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1] + [0] * 11
    for cut in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:cut], indicators.iloc[:cut], None, params).data
        assert prefix['signal'].equals(result['signal'].iloc[:cut])
    flat = data.copy()
    flat['close'] = 100.0
    assert not strategy.indicators(flat, params)['entry'].any()
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= params.regime_bars + 1


if __name__ == '__main__':
    _self_check()
