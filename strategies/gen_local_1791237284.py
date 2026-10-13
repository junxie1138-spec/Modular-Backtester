from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    trend_window: int = 20
    innovation_window: int = 4
    innovation_threshold: float = 0.6
    atr_window: int = 14
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = 'gen_local_1791237284'

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return int(max(200, params.trend_window + params.innovation_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        if params.trend_window < 2 or min(params.innovation_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive; trend_window must be at least two.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError('ATR multiplier and innovation threshold must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0.0, np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        n = params.trend_window
        midpoint = (n - 1) / 2.0
        out['mean_close'] = close.rolling(n, min_periods=n).mean()
        weighted = sum((j - midpoint) * close.shift(n - 1 - j) for j in range(n))
        out['slope'] = weighted / (n * (n * n - 1) / 12.0)
        out['level'] = out['mean_close'] + midpoint * out['slope']
        out['forecast'] = (out['level'] + out['slope']).shift(1)
        out['innovation'] = close - out['forecast']
        out['score'] = out['innovation'].rolling(params.innovation_window, min_periods=params.innovation_window).mean() / out['atr']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: InnovationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df['signal'] = 0
        df['size'] = 1.0
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        slope = indicators['slope'].to_numpy(dtype=float)
        score = indicators['score'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            valid = np.isfinite([close[i], opens[i], atr[i], slope[i], score[i], regime[i]]).all() and atr[i] > 0
            if not valid:
                position = 0
                continue
            if position:
                # The decision on the previous bar fills at this bar's open.
                if i == entry_bar:
                    extreme = opens[i]
                    stop = extreme - position * params.trail_atr * atr[i]
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                    regime_exit = close[i] <= regime[i]
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                    regime_exit = close[i] >= regime[i]
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or regime_exit or expired:
                    position = 0
            else:
                previous_score = score[i - 1] if i > 0 else np.nan
                if np.isfinite(previous_score):
                    if close[i] > regime[i] and slope[i] > 0 and score[i] >= params.innovation_threshold and previous_score < params.innovation_threshold:
                        position = 1
                    elif close[i] < regime[i] and slope[i] < 0 and score[i] <= -params.innovation_threshold and previous_score > -params.innovation_threshold:
                        position = -1
                if position:
                    entry_bar = i + 1
                    extreme = np.nan
                    stop = np.nan
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = InnovationParams()
    index = pd.date_range('2020-01-01', periods=7)
    for direction in (1, -1):
        closes = 100.0 + direction * np.array([1, 1, 3, 4, 2, 0, 0], dtype=float)
        data = pd.DataFrame({'open': np.full(7, 100.0), 'high': closes + 1, 'low': closes - 1, 'close': closes, 'volume': np.full(7, 1000.0)}, index=index)
        indicators = pd.DataFrame({'atr': [1, 1, 1, 1, 5, 5, 5], 'slope': np.full(7, direction), 'score': direction * np.array([0, 1, 1, 1, 1, 1, 1]), 'ma200': np.full(7, 100.0 - 10 * direction)}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, direction, 0]
        assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
