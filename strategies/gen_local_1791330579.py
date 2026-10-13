from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnDebtParams:
    breakout_window: int = 20
    debt_window: int = 3
    debt_multiple: float = 1.0
    atr_window: int = 14
    trail_atr: float = 1.0
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReturnDebtParams]):
    strategy_id = 'gen_local_1791330579'

    @classmethod
    def params_type(cls) -> type[ReturnDebtParams]:
        return ReturnDebtParams

    def warmup_bars(self, params: ReturnDebtParams) -> int:
        return int(max(200, params.breakout_window + 2,
                       params.debt_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnDebtParams) -> pd.DataFrame:
        if min(params.breakout_window, params.debt_window, params.atr_window) < 1:
            raise ValueError('Lookback windows must be positive.')
        if params.debt_multiple <= 0 or params.trail_atr <= 0:
            raise ValueError('Debt and ATR multipliers must be positive.')
        if params.holding_bars not in (1, 2):
            raise ValueError('holding_bars must be 1 or 2.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['downside_return'] = -out['return'].clip(upper=0)
        out['downside_debt'] = out['downside_return'].rolling(params.debt_window).sum().shift(1)
        out['prior_net_return'] = out['return'].rolling(params.debt_window).sum().shift(1)
        out['prior_return_record'] = out['return'].rolling(params.breakout_window).max().shift(1)
        out['ma200'] = close.rolling(200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        ready = np.isfinite(out).all(axis=1) & (out['atr'] > 0)
        out['entry'] = (
            ready
            & (close > out['ma200'])
            & (out['prior_net_return'] < 0)
            & (out['downside_debt'] > 0)
            & (out['return'] > 0)
            & (out['return'] > out['prior_return_record'])
            & (out['return'] > params.debt_multiple * out['downside_debt'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ReturnDebtParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest = np.nan
        stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if not (np.isfinite(closes[i]) and np.isfinite(opens[i])
                        and np.isfinite(atrs[i]) and atrs[i] > 0):
                    active = False
                    continue
                if age == 1:
                    # The preceding close's decision fills at this bar's open.
                    highest = opens[i]
                    stop = highest - params.trail_atr * atrs[i - 1]
                highest = max(highest, closes[i])
                stop = max(stop, highest - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.holding_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                age = 0
                highest = np.nan
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 102.0, 100.0],
        'high': [101.0, 103.0, 103.0, 101.0],
        'low': [99.0, 99.0, 99.0, 99.0],
        'close': [100.0, 102.0, 100.0, 100.0],
        'volume': [1000.0] * 4
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 10.0, 1.0],
        'entry': [True, False, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    params = ReturnDebtParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    # Isolate the trail: widening ATR must not lower an existing stop.
    indicators['atr'] = [1.0, 10.0, 1.0, 1.0]
    data['close'] = [100.0, 98.5, 100.0, 100.0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    params.holding_bars = 1
    indicators['atr'] = 1.0
    data['close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    assert not strategy.indicators(data, params)['entry'].any()
    assert strategy.warmup_bars(params) >= 200
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
