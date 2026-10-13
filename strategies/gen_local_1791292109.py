from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectedHighParams:
    rejection_bars: int = 3
    atr_period: int = 14
    setup_window: int = 10
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RejectedHighParams]):
    strategy_id = 'gen_local_1791292109'

    @classmethod
    def params_type(cls) -> type[RejectedHighParams]:
        return RejectedHighParams

    def warmup_bars(self, params: RejectedHighParams) -> int:
        return int(params.atr_period + params.rejection_bars + params.setup_window + 2)

    def indicators(self, data: pd.DataFrame, params: RejectedHighParams) -> pd.DataFrame:
        if min(params.rejection_bars, params.atr_period, params.setup_window, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        previous_high = data['high'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out['rejected'] = ((data['high'] > previous_high) & (data['close'] <= previous_high)).fillna(False)
        out['rejection_streak'] = out['rejected'].astype(int).groupby((~out['rejected']).cumsum()).cumsum()
        out['bar'] = np.arange(len(data), dtype=float)
        setup = out['rejection_streak'] >= params.rejection_bars
        out['setup_bar'] = out['bar'].where(setup).ffill()
        out['ceiling'] = data['high'].where(setup).ffill()
        out['setup_age'] = out['bar'] - out['setup_bar']
        out['breakout'] = (
            (data['close'] > out['ceiling'])
            & out['setup_age'].between(1, params.setup_window)
            & np.isfinite(out['atr'])
            & out['atr'].gt(0)
        ).fillna(False)
        out['confirmed'] = out['breakout'] & out['breakout'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RejectedHighParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        setups = indicators['setup_bar'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        held = False
        entry_bar = -1
        used_setup = -1
        stop = np.nan
        for i in range(len(data)):
            if held:
                if i == entry_bar:
                    # The previous close's decision fills at this bar's open.
                    # Freeze both the entry price and the decision-bar ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                expired = i - entry_bar + 1 >= params.hold_bars
                stopped = np.isfinite(closes[i]) and closes[i] < stop
                if not np.isfinite(stop) or expired or stopped:
                    held = False
                else:
                    raw[i] = 1
                continue
            if confirmed[i] and np.isfinite(setups[i]) and int(setups[i]) != used_setup:
                held = True
                entry_bar = i + 1
                used_setup = int(setups[i])
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=11, freq='B')
    close = np.array([100., 100., 101., 102., 105., 106., 107., 110., 105., 103., 104.])
    data = pd.DataFrame({
        'open': close,
        'high': [101., 102., 103., 104., 106., 107., 108., 111., 106., 104., 105.],
        'low': close - 1.0,
        'close': close,
        'volume': 1000000.0,
    }, index=index)
    params = RejectedHighParams(atr_period=3, stop_atr=1.0)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert derived['rejection_streak'].iloc[1:4].tolist() == [1, 2, 3]
    assert not derived['confirmed'].iloc[4] and derived['confirmed'].iloc[5]
    # Entry opens at 107 with frozen ATR 3: stop stays 104 after close reaches 110.
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
