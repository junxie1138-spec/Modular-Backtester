from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_window: int = 20
    balance_window: int = 4
    volume_multiple: float = 1.3
    balance_threshold: float = 0.4
    profit_target: float = 0.025
    max_hold_bars: int = 10
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790851682'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(205, params.volume_window + 1, params.balance_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.volume_window < 1 or params.balance_window < 1:
            raise ValueError('Lookbacks must be positive')
        if params.volume_multiple <= 1 or not 0 < params.balance_threshold <= 1:
            raise ValueError('Invalid volume thresholds')
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['ma_slope'] = out['ma200'].diff(5)
        out['change'] = close.diff()
        out['volume_base'] = volume.rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out['relative_volume'] = volume / out['volume_base'].replace(0, np.nan)
        out['signed_volume'] = np.sign(out['change']) * volume
        out['volume_total'] = volume.rolling(params.balance_window, min_periods=params.balance_window).sum()
        out['volume_balance'] = out['signed_volume'].rolling(params.balance_window, min_periods=params.balance_window).sum() / out['volume_total'].replace(0, np.nan)
        out['prior_midpoint'] = ((data['high'] + data['low']) / 2).shift(1)
        out['above_ma'] = close > out['ma200']
        out['rising_regime'] = out['ma_slope'] > 0
        out['accumulation'] = (out['change'] > 0) & (out['relative_volume'] >= params.volume_multiple) & (out['volume_balance'] >= params.balance_threshold)
        out['distribution_spike'] = (out['change'] < 0) & (out['relative_volume'] >= params.volume_multiple)
        out['absorption'] = out['distribution_spike'].shift(1, fill_value=False) & (out['change'] > 0) & (out['relative_volume'] < 1.0) & (close > out['prior_midpoint'])
        out['ready'] = np.isfinite(out[['ma200', 'ma_slope', 'change', 'relative_volume', 'volume_balance', 'prior_midpoint']]).all(axis=1)
        out['entry'] = (out['ready'] & out['above_ma'] & ((out['rising_regime'] & out['accumulation']) | (~out['rising_regime'] & out['absorption']))).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars < 1 or params.cooldown_bars < 0:
            raise ValueError('Invalid exit parameters')
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        next_entry = 0
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Observe the entry open only when its bar is reached.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    next_entry = i + params.cooldown_bars
                elif closes[i] >= entry_price * (1 + params.profit_target) or i - entry_bar + 1 >= params.max_hold_bars:
                    # Close-based exit decision fills at the following open.
                    active = False
                    next_entry = i + params.cooldown_bars
            elif i >= next_entry and entries[i]:
                active = True
                entry_bar = i + 1
            signal[i] = int(active)
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry': True}, index=index)
    params = Params(max_hold_bars=3, cooldown_bars=2)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 1, 1]
    assert result['size'].gt(0).all()
    entries['entry'] = [True] + [False] * 7
    data.loc[index[1], 'close'] = 103.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    data.loc[index[1], 'open'] = 110.0
    data.loc[index[1:], 'close'] = 110.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert not strategy.indicators(data, params)['entry'].any()
    assert strategy.warmup_bars(params) >= 205


if __name__ == '__main__':
    _self_check()
