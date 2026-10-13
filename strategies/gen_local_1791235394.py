from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    envelope_bars: int = 12
    percentile_bars: int = 60
    recoil_quantile: float = 0.80
    atr_bars: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = 'gen_local_1791235394'

    @classmethod
    def params_type(cls):
        return RecoilParams

    @staticmethod
    def warmup_bars(params: RecoilParams) -> int:
        return params.envelope_bars + params.percentile_bars + params.atr_bars + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        if min(params.envelope_bars, params.percentile_bars, params.atr_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.0 < params.recoil_quantile < 1.0:
            raise ValueError('recoil_quantile must lie between zero and one')
        if params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError('stop_atr and hold_bars must be positive')
        out = pd.DataFrame(index=data.index)
        high, low, close = data['high'], data['low'], data['close']
        out['width'] = (high - low).where(high > low)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            high - low,
            (high - out['previous_close']).abs(),
            (low - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['floor'] = low.rolling(params.envelope_bars, min_periods=params.envelope_bars).min().shift(1)
        out['ceiling'] = high.rolling(params.envelope_bars, min_periods=params.envelope_bars).max().shift(1)
        out['envelope_width'] = (out['ceiling'] - out['floor']).where(out['ceiling'] > out['floor'])
        out['position'] = (close - out['floor']) / out['envelope_width']
        out['position_median'] = out['position'].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(0.5).shift(1)
        out['floor_lift'] = low.diff()
        out['ceiling_retreat'] = -high.diff()
        out['recoil'] = (out['floor_lift'] - out['ceiling_retreat']) / out['width'].shift(1)
        out['recoil_threshold'] = out['recoil'].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.recoil_quantile).shift(1)
        out['probe'] = (low < out['floor']).fillna(False)
        out['previous_midpoint'] = ((high + low) / 2.0).shift(1)
        ready = np.isfinite(out[['atr', 'position', 'position_median', 'recoil', 'recoil_threshold', 'previous_midpoint']]).all(axis=1)
        out['setup'] = (
            ready
            & out['probe'].shift(1).fillna(False).astype(bool)
            & (out['floor_lift'] > 0)
            & (out['ceiling_retreat'] > 0)
            & (out['recoil'] > out['recoil_threshold'])
            & (close > out['previous_midpoint'])
            & (out['position'] < out['position_median'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoilParams) -> SignalFrame:
        if params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError('stop_atr and hold_bars must be positive')
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            # A decision on the previous close enters at this bar's open.
            if not active and i > 0 and raw[i - 1] == 1:
                price = data['open'].iat[i]
                atr = indicators['atr'].iat[i - 1]
                if np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0:
                    active = True
                    entry_bar = i
                    # ponytail: nominal opening fill; use execution feedback if exact slippage-adjusted stops become necessary.
                    stop = price - params.stop_atr * atr
                else:
                    continue
            if active:
                close = data['close'].iat[i]
                if not np.isfinite(close) or close < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif indicators['setup'].iat[i]:
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'close': [100.0, 100.0, 102.0, 97.0, 101.0, 101.0, 101.0]}, index=index)
    ind = pd.DataFrame({'atr': [2.0, 2.0, 20.0, 20.0, 20.0, 20.0, 20.0], 'setup': [True, False, False, False, False, False, False]}, index=index)
    params = RecoilParams(stop_atr=1.0, hold_bars=4)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data['close'] = 100.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
