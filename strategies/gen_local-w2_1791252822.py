from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 60
    occupancy_window: int = 20
    trough_window: int = 10
    rank_window: int = 126
    occupancy_percentile: float = 0.80
    recovery_percentile: float = 0.80
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791252822'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(
            params.peak_window + params.occupancy_window + params.rank_window,
            max(params.peak_window, params.trough_window) + params.rank_window + 1,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        windows = (params.peak_window, params.occupancy_window,
                   params.trough_window, params.rank_window, params.atr_window)
        if any(w < 2 for w in windows) or params.hold_bars < 1:
            raise ValueError('Windows must be at least two and hold_bars positive.')
        if not (0 < params.occupancy_percentile <= 1 and
                0 < params.recovery_percentile <= 1 and
                np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError('Invalid percentile or stop multiplier.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['peak'] = close.rolling(params.peak_window).max()
        out['trough'] = close.rolling(params.trough_window).min()
        out['drawdown'] = close / out['peak'].where(out['peak'] > 0) - 1.0
        out['underwater'] = (close < out['peak']).astype(float).where(out['peak'].notna())
        out['occupancy'] = out['underwater'].rolling(params.occupancy_window).sum()
        out['occupancy_rank'] = out['occupancy'].rolling(params.rank_window).rank(pct=True, method='max')
        span = out['peak'] - out['trough']
        out['recovery_share'] = (close - out['trough']) / span.where(span > 0)
        out['recovery_rank'] = out['recovery_share'].rolling(params.rank_window).rank(pct=True)
        out['previous_recovery_rank'] = out['recovery_rank'].shift(1)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['entry'] = (
            (out['drawdown'] < 0)
            & (out['occupancy_rank'] >= params.occupancy_percentile)
            & (out['recovery_rank'] >= params.recovery_percentile)
            & (out['previous_recovery_rank'] < params.recovery_percentile)
            & np.isfinite(out['atr'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                # Freeze the stop at the next-bar entry open using decision-bar ATR.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                held = 0
                active = True
            if active:
                held += 1
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or held >= params.hold_bars):
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100, 100, 100, 110, 100, 100],
        'close': [100, 101, 110, 95, 100, 100],
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2, 2, 20, 20, 2, 2],
        'entry': [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    data['close'] = 101.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, RecoveryParams(hold_bars=2)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    indicators['entry'] = False
    result = GeneratedStrategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
