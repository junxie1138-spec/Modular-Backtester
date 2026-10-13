from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 40
    trough_window: int = 10
    volatility_window: int = 5
    minimum_drawdown: float = 0.03
    repair_min: float = 0.20
    repair_max: float = 0.55
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        if min(self.peak_window, self.trough_window, self.volatility_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if self.trough_window > self.peak_window:
            raise ValueError('Trough window must not exceed peak window')
        if not 0 < self.minimum_drawdown < 1:
            raise ValueError('Minimum drawdown must be between zero and one')
        if not 0 <= self.repair_min < self.repair_max < 1:
            raise ValueError('Invalid repair bounds')
        if self.profit_target <= 0 or self.max_hold_bars < 1:
            raise ValueError('Exit parameters must be positive')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791236202'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.trough_window,
                       params.volatility_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['up_energy'] = out['return'].clip(lower=0).pow(2)
        out['down_energy'] = out['return'].clip(upper=0).pow(2)
        window = params.volatility_window
        out['up_volatility'] = np.sqrt(out['up_energy'].rolling(window, min_periods=window).mean())
        out['down_volatility'] = np.sqrt(out['down_energy'].rolling(window, min_periods=window).mean())
        out['volatility_balance'] = out['up_volatility'] - out['down_volatility']
        out['previous_balance'] = out['volatility_balance'].shift(1)
        out['volatility_agrees'] = ((out['previous_balance'] < 0) &
                                    (out['volatility_balance'] > 0))
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['trough'] = close.rolling(params.trough_window, min_periods=params.trough_window).min()
        out['depth'] = 1.0 - out['trough'] / out['peak']
        out['repair'] = (close - out['trough']) / (out['peak'] - out['trough']).replace(0, np.nan)
        out['price_agrees'] = ((out['depth'] >= params.minimum_drawdown) &
                              (out['repair'] >= params.repair_min) &
                              (out['repair'] <= params.repair_max))
        out['entry'] = (out['volatility_agrees'] & out['price_agrees']).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                # The previous close's decision fills at this bar's open.
                entry_price = opens[i]
                active = bool(np.isfinite(entry_price) and entry_price > 0)
                age = 0
                pending = False
                if not active:
                    continue
            if active:
                age += 1
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.max_hold_bars or not np.isfinite(closes[i]):
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': [100, 110, 110, 110, 110, 110],
                         'high': [101, 111, 113, 115, 115, 115],
                         'low': [99, 109, 109, 109, 109, 109],
                         'close': [100, 110, 112, 114, 114, 114],
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    target = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert target['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    timed = strategy.generate_signals(data, indicators, None,
                                      RecoveryParams(profit_target=0.5, max_hold_bars=2)).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    warmup = strategy.indicators(data, RecoveryParams())
    assert not warmup['entry'].any()
    assert (target['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
