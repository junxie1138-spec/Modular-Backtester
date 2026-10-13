from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    reference_window: int = 20
    reaction_window: int = 5
    atr_window: int = 14
    change_threshold: float = 1.5
    volume_confirmation: float = 1.15
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = 'gen_local_1791267275'

    @classmethod
    def params_type(cls) -> type[ChangeParams]:
        return ChangeParams

    def warmup_bars(self, params: ChangeParams) -> int:
        return int(max(200, 2 * params.reference_window + params.reaction_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        if min(params.reference_window, params.reaction_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if min(params.change_threshold, params.volume_confirmation, params.trail_atr) <= 0 or params.holding_bars < 1:
            raise ValueError('Thresholds and holding duration must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        volume = data['volume'].where(data['volume'] >= 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return'] = close.pct_change(fill_method=None)
        out['volume_base'] = volume.shift(1).rolling(params.reference_window, min_periods=params.reference_window).mean()
        out['relative_volume'] = volume.div(out['volume_base'].where(out['volume_base'] > 0)).clip(upper=5.0)
        out['pressure'] = out['return'] * out['relative_volume']
        out['recent_pressure'] = out['pressure'].rolling(params.reaction_window, min_periods=params.reaction_window).mean()
        out['old_pressure'] = out['pressure'].shift(params.reaction_window).rolling(params.reference_window, min_periods=params.reference_window).mean()
        out['old_deviation'] = out['pressure'].shift(params.reaction_window).rolling(params.reference_window, min_periods=params.reference_window).std(ddof=0)
        out['change_score'] = (out['recent_pressure'] - out['old_pressure']).div(out['old_deviation'].where(out['old_deviation'] > 0)) * np.sqrt(params.reaction_window)
        out['recent_volume'] = volume.rolling(params.reaction_window, min_periods=params.reaction_window).mean()
        out['old_volume'] = volume.shift(params.reaction_window).rolling(params.reference_window, min_periods=params.reference_window).mean()
        out['participation'] = out['recent_volume'].div(out['old_volume'].where(out['old_volume'] > 0))
        out['recent_return'] = close.pct_change(periods=params.reaction_window, fill_method=None)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ChangeParams) -> SignalFrame:
        score = indicators['change_score']
        confirmed = indicators['participation'].ge(params.volume_confirmation)
        long_entry = (confirmed & score.ge(params.change_threshold) & score.shift(1).lt(params.change_threshold) & indicators['recent_return'].gt(0) & data['close'].gt(indicators['ma200'])).fillna(False).to_numpy(dtype=bool)
        short_entry = (confirmed & score.le(-params.change_threshold) & score.shift(1).gt(-params.change_threshold) & indicators['recent_return'].lt(0) & data['close'].lt(indicators['ma200'])).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        regimes = indicators['ma200'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            ready = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(regimes[i])
            if not ready:
                position = 0
                age = 0
                watermark = stop = -np.inf
                continue
            if position:
                # Signed prices give a high-close trail for longs and its low-close mirror for shorts.
                if age == 0 and np.isfinite(opens[i]) and opens[i] > 0:
                    watermark = position * opens[i]
                age += 1
                favorable_close = position * closes[i]
                watermark = max(watermark, favorable_close)
                stop = max(stop, watermark - params.trail_atr * atrs[i])
                regime_lost = position * (closes[i] - regimes[i]) <= 0
                if favorable_close <= stop or age >= params.holding_bars or regime_lost:
                    position = 0
                    age = 0
                    watermark = stop = -np.inf
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
                watermark = stop = -np.inf
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': [100, 100, 101, 104, 102, 100, 99, 95], 'high': [101, 102, 105, 105, 103, 101, 100, 99], 'low': [99, 99, 100, 101, 99, 98, 94, 94], 'close': [100, 101, 104, 102, 100, 99, 95, 98], 'volume': np.full(8, 1000.0)}, index=index)
    indicators = pd.DataFrame({'change_score': [0, 2, 2, 2, 0, -2, -2, -2], 'participation': np.full(8, 2.0), 'recent_return': [0, 0.01, 0.02, 0.01, 0, -0.01, -0.02, -0.01], 'ma200': [90, 90, 90, 90, 90, 110, 110, 110], 'atr': [1, 1, 1, 2, 1, 1, 1, 1]}, index=index)
    params = ChangeParams(trail_atr=2.0)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, -1, -1]
    assert result.index.equals(index) and result['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(None, params) >= 200


if __name__ == '__main__':
    _self_check()
