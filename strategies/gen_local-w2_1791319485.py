from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_lookback: int = 20
    absorption_lookback: int = 3
    gap_multiple: float = 1.2
    wick_threshold: float = 0.15
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791319485'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if params.gap_lookback < 2 or params.absorption_lookback < 1:
            raise ValueError('Lookbacks must be positive; gap_lookback must be at least two.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        if not (np.isfinite(params.gap_multiple) and params.gap_multiple > 0):
            raise ValueError('gap_multiple must be finite and positive.')
        if not (np.isfinite(params.wick_threshold) and 0 < params.wick_threshold < 1):
            raise ValueError('wick_threshold must be between zero and one.')
        if not (np.isfinite(params.profit_target) and 0 < params.profit_target < 1):
            raise ValueError('profit_target must be between zero and one.')
        return int(max(params.gap_lookback + 2, params.absorption_lookback + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        prior_close = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = (data['open'] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        out['gap_scale'] = out['gap'].abs().rolling(params.gap_lookback, min_periods=params.gap_lookback).mean().shift(1)
        out['gap_strength'] = out['gap'] / out['gap_scale'].where(out['gap_scale'] > 0)
        body_low = data[['open', 'close']].min(axis=1)
        body_high = data[['open', 'close']].max(axis=1)
        out['lower_wick'] = (body_low - data['low']).clip(lower=0)
        out['upper_wick'] = (data['high'] - body_high).clip(lower=0)
        out['bar_range'] = data['high'] - data['low']
        out['wick_balance'] = (out['lower_wick'] - out['upper_wick']) / out['bar_range'].where(out['bar_range'] > 0)
        out['absorption'] = out['wick_balance'].rolling(params.absorption_lookback, min_periods=params.absorption_lookback).mean().shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        bullish = (indicators['gap_strength'] >= params.gap_multiple) & (indicators['absorption'] >= params.wick_threshold)
        bearish = (indicators['gap_strength'] <= -params.gap_multiple) & (indicators['absorption'] <= -params.wick_threshold)
        entries = np.select([bullish.fillna(False), bearish.fillna(False)], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            valid = np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0
            if side:
                # An entry decided at the preceding close fills at this open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                gain = side * (closes[i] / entry_price - 1) if valid and np.isfinite(entry_price) and entry_price > 0 else np.nan
                if not valid or not np.isfinite(gain) or gain >= params.profit_target or held >= params.hold_bars:
                    side = 0
                    entry_price = np.nan
                    held = 0
            elif valid and entries[i] != 0:
                side = int(entries[i])
                held = 0
            desired[i] = side
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'gap_strength': [2.0] + [0.0] * 7, 'absorption': [0.3] + [0.0] * 7}, index=index)
    params = Params()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], 'close'] = 102.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    indicators.loc[index[0], ['gap_strength', 'absorption']] = [-2.0, -0.3]
    data.loc[index[2], 'close'] = 98.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0, 0, 0, 0]
    indicators.loc[index[0], 'absorption'] = 0.3
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
