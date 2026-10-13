from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SweepParams:
    lookback: int = 60
    gap_quantile: float = 0.15
    floor_quantile: float = 0.20
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SweepParams]):
    strategy_id = 'gen_local_1790860974'

    @classmethod
    def params_type(cls):
        return SweepParams

    def warmup_bars(self, params: SweepParams) -> int:
        if params.lookback < 2 or params.atr_period < 1 or params.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0.0 < params.gap_quantile < 0.5 and 0.0 < params.floor_quantile < 0.5):
            raise ValueError('Quantiles must be between zero and 0.5.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: SweepParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        out['gap'] = data['open'].div(out['prior_close'].where(out['prior_close'] > 0)).sub(1.0)
        out['gap_cutoff'] = out['gap'].rolling(params.lookback, min_periods=params.lookback).quantile(params.gap_quantile).shift(1)
        out['floor'] = data['close'].rolling(params.lookback, min_periods=params.lookback).quantile(params.floor_quantile).shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['prior_close']).abs(),
            (data['low'] - out['prior_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        ready = np.isfinite(out[['prior_close', 'gap', 'gap_cutoff', 'floor', 'atr']]).all(axis=1)
        ready &= np.isfinite(data[['open', 'close']]).all(axis=1)
        out['entry'] = (
            ready
            & (out['atr'] > 0)
            & (data['open'] > 0)
            & (data['close'] > 0)
            & (out['gap'] < 0)
            & (out['gap'] <= out['gap_cutoff'])
            & (out['prior_close'] >= out['floor'])
            & (data['open'] < out['floor'])
            & (data['close'] > out['floor'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SweepParams) -> SignalFrame:
        self.warmup_bars(params)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = 0.0
        stop = 0.0
        entry_bar = 0

        for i in range(len(data)):
            # A prior close decision fills at this bar's open.
            if pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError('Entry open must be finite and positive.')
                stop = opens[i] - params.stop_atr * entry_atr
                entry_bar = i
                active = True
                pending = False

            if active:
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    raise ValueError('Held-position close must be finite and positive.')
                # The stop stays fixed at entry open minus signal-bar ATR.
                if closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = True
                entry_atr = atrs[i]

            signals[i] = int(active or pending)

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=17, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 112.0, 'low': 94.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 2.0, 'entry': False}, index=index)
    indicators.loc[index[1], 'entry'] = True
    indicators.loc[index[6], 'entry'] = True
    data.loc[index[2], 'close'] = 110.0
    data.loc[index[3], 'close'] = 98.0
    data.loc[index[4], 'close'] = 95.0
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, SweepParams()).data
    expected = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert result['signal'].tolist() == expected
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(SweepParams()) == 62
    warmup = strategy.indicators(data, SweepParams())
    assert not warmup['entry'].any()


if __name__ == '__main__':
    _self_check()
