from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreathingParams:
    atr_window: int = 10
    volatility_window: int = 20
    impulse_threshold: float = 0.15
    target_daily_vol: float = 0.0075
    max_size: float = 0.95

    def __post_init__(self):
        if self.atr_window < 2 or self.volatility_window < 2:
            raise ValueError('Lookbacks must be at least two bars')
        values = (self.impulse_threshold, self.target_daily_vol, self.max_size)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Threshold, volatility target and size must be finite and positive')
        if self.max_size > 1:
            raise ValueError('max_size must not exceed one')


class GeneratedStrategy(BaseStrategy[BreathingParams]):
    strategy_id = 'gen_local_1791316023'

    @classmethod
    def params_type(cls) -> type[BreathingParams]:
        return BreathingParams

    def warmup_bars(self, params: BreathingParams) -> int:
        return int(max(params.atr_window + 2, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: BreathingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['previous_atr'] = out['atr'].shift(1)
        out['range_change'] = out['true_range'].diff()
        out['candle_commitment'] = (
            (data['close'] - data['open']) / out['true_range'].where(out['true_range'] > 0)
        ).clip(-1.0, 1.0)
        out['impulse'] = (
            out['range_change'] / out['previous_atr'].where(out['previous_atr'] > 0)
        ) * out['candle_commitment']
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BreathingParams,
    ) -> SignalFrame:
        impulse = indicators['impulse'].replace([np.inf, -np.inf], np.nan)
        volatility = indicators['volatility'].replace([np.inf, -np.inf], np.nan)
        ready = impulse.notna() & volatility.notna() & (volatility > 0)
        entries = pd.Series(0, index=data.index, dtype=int)
        entries.loc[ready & (impulse >= params.impulse_threshold)] = 1
        entries.loc[ready & (impulse <= -params.impulse_threshold)] = -1

        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries.to_numpy()):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # The opposite entry condition exits; a later bar may re-enter.
                position = 0
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        minimum_size = min(0.05, params.max_size)
        decision_size = (
            params.target_daily_vol / volatility.where(volatility > 0)
        ).clip(lower=minimum_size, upper=params.max_size).fillna(minimum_size)
        df['size'] = decision_size.shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = BreathingParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'impulse': [1.0, 0.0, -1.0, -1.0, np.nan, 1.0, 1.0],
        'volatility': [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.isclose(result['size'].iloc[2], 0.375)
    assert (result['size'] > 0).all()
    indicators['impulse'] *= -1
    mirrored = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert mirrored['signal'].equals(-result['signal'])
