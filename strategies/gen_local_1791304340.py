from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 6
    area_threshold: float = 0.06
    min_efficiency: float = 0.25
    atr_window: int = 14
    stop_atr: float = 0.9
    hold_bars: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791304340'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        overnight = np.log(opening / close.shift(1))
        intraday = np.log(close / opening)
        result = pd.DataFrame(index=data.index)
        result['overnight'] = overnight
        result['intraday'] = intraday
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.window):
            cross = overnight.shift(lag) * intraday - intraday.shift(lag) * overnight
            area = area + 0.5 * cross.rolling(params.window - lag).sum()
        result['signature_area'] = area
        result['overnight_variation'] = overnight.abs().rolling(params.window).sum()
        result['intraday_variation'] = intraday.abs().rolling(params.window).sum()
        denominator = result['overnight_variation'] * result['intraday_variation']
        result['normalized_area'] = (area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        result['drift'] = (overnight + intraday).rolling(params.window).sum()
        result['path_variation'] = (overnight.abs() + intraday.abs()).rolling(params.window).sum()
        result['efficiency'] = (result['drift'].abs() / result['path_variation'].where(result['path_variation'] > 0)).clip(0.0, 1.0)
        previous = data['close'].shift(1)
        result['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        result['atr'] = result['true_range'].rolling(params.atr_window).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        area = indicators['normalized_area']
        efficiency = indicators['efficiency']
        drift = indicators['drift']
        atr = indicators['atr']
        valid = pd.concat([area, efficiency, drift, atr, data['close']], axis=1).apply(np.isfinite).all(axis=1) & (atr > 0)
        eligible = valid & (area > params.area_threshold) & (efficiency >= params.min_efficiency) & (drift != 0)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & (drift > 0)] = 1
        candidates.loc[eligible & (drift < 0)] = -1
        strength = ((area / params.area_threshold - 1.0).clip(0.0, 1.0) * efficiency).fillna(0.0)
        candidate_size = (params.min_size + (params.max_size - params.min_size) * strength).clip(params.min_size, params.max_size)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        age = 0
        highest = -np.inf
        trailing_stop = -np.inf
        position_size = params.min_size
        for i in range(len(data)):
            if position != 0:
                age += 1
                if not bool(valid.iloc[i]):
                    position = 0
                else:
                    # Entry fills on this bar or an earlier bar, so only live-trade closes enter the watermark.
                    highest = max(highest, float(data['close'].iloc[i]))
                    trailing_stop = max(trailing_stop, highest - params.stop_atr * float(atr.iloc[i]))
                    # The required close-high drawdown exit applies to both directions.
                    if float(data['close'].iloc[i]) <= trailing_stop or age >= params.hold_bars:
                        position = 0
                signals[i] = position
                sizes[i] = position_size if position else params.min_size
                continue
            if candidates.iloc[i] != 0:
                position = int(candidates.iloc[i])
                position_size = float(candidate_size.iloc[i])
                age = 0
                highest = -np.inf
                trailing_stop = -np.inf
            signals[i] = position
            sizes[i] = position_size if position else params.min_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [100.0] * 5, 'high': [102.0] * 5, 'low': [98.0] * 5, 'close': [100.0, 101.0, 99.0, 100.0, 100.0], 'volume': [1000.0] * 5}, index=index)
    indicators = pd.DataFrame({'normalized_area': [0.12, 0.0, 0.0, 0.0, 0.0], 'efficiency': [0.8] * 5, 'drift': [0.01] * 5, 'atr': [1.0] * 5}, index=index)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        indicators['drift'] = direction * 0.01
        result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]
        assert result['size'].between(0.25, 0.95).all()
        assert result['size'].iloc[1] == result['size'].iloc[2] > 0.25
    data['close'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0]
    data['close'] = [100.0, 101.0, 99.0, 100.0, 100.0]
    result = strategy.generate_signals(data, indicators, None, SignatureParams(hold_bars=10)).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0]


if __name__ == '__main__':
    _self_check()
