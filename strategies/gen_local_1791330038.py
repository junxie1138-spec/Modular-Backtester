from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    history: int = 252
    volume_window: int = 20
    atr_window: int = 14
    bandwidth: float = 0.35
    forecast_floor: float = 0.001
    stop_atr: float = 0.9


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791330038'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.history + max(params.volume_window, params.atr_window) + 3)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if params.history < 10 or params.volume_window < 2 or params.atr_window < 2:
            raise ValueError('Lookback parameters are too small')
        if not np.isfinite(params.bandwidth) or params.bandwidth <= 0:
            raise ValueError('bandwidth must be finite and positive')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not np.isfinite(params.forecast_floor) or params.forecast_floor < 0:
            raise ValueError('forecast_floor must be finite and nonnegative')

        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['volume_baseline'] = data['volume'].shift(1).rolling(params.volume_window, min_periods=params.volume_window).mean()
        out['volume_ratio'] = data['volume'] / out['volume_baseline'].where(out['volume_baseline'] > 0)
        candle_range = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['body'] = (data['close'] - data['open']) / candle_range
        out['log_volume'] = np.log(out['volume_ratio'].where(out['volume_ratio'] > 0)).clip(-2.0, 2.0)
        out['return'] = data['close'] / previous_close.where(previous_close > 0) - 1.0
        features = ['body', 'log_volume']
        for lag in (1, 2):
            for name in ('body', 'log_volume'):
                column = f'{name}_{lag}'
                out[column] = out[name].shift(lag)
                features.append(column)
        out = out.replace([np.inf, -np.inf], np.nan)

        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        # Each historical outcome is fully observed before the current pattern.
        # ponytail: O(bars * history), use a bounded nearest-neighbor search if history becomes large.
        for lag in range(5, params.history + 1):
            distance = pd.Series(0.0, index=data.index)
            for name in features:
                distance = distance + (out[name] - out[name].shift(lag)).pow(2)
            distance = distance / len(features)
            historical_open = data['open'].shift(lag - 1)
            outcome = data['close'].shift(lag - 2) / historical_open.where(historical_open > 0) - 1.0
            outcome = outcome.replace([np.inf, -np.inf], np.nan)
            weight = np.exp(-distance / (2.0 * params.bandwidth ** 2))
            weight = weight.where(outcome.notna(), 0.0).fillna(0.0)
            numerator = numerator + weight * outcome.fillna(0.0)
            denominator = denominator + weight
        out['analog_weight'] = denominator
        out['forecast'] = numerator / denominator.where(denominator > 0)
        out['volume_confirmed'] = (
            out['return'].gt(0)
            & out['body'].gt(0)
            & out['volume_ratio'].ge(1.05)
        )
        out['entry_ready'] = (
            out['volume_confirmed']
            & out['volume_confirmed'].shift(1, fill_value=False)
            & out['body_2'].lt(0)
            & out['forecast'].gt(params.forecast_floor)
            & out['analog_weight'].ge(6.0)
            & out['atr'].gt(0)
        )
        out.loc[out.index[:self.warmup_bars(params)], 'entry_ready'] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        ready = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        pending = False
        age = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                # Entry executes on this bar's open; ATR was fixed at the decision close.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                active = True
                age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= 2:
                    active = False
                else:
                    raw[i] = 1
            elif ready[i] and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    sample = pd.DataFrame({
        'open': [100.0] * 8,
        'high': [101.0] * 8,
        'low': [97.0] * 8,
        'close': [100.0, 100.0, 100.0, 98.0, 100.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 8,
    }, index=index)
    derived = pd.DataFrame({
        'atr': [1.0] * 8,
        'entry_ready': [False, False, True, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, ShapeParams()).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
