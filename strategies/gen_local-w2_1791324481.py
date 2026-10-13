from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactStreakParams:
    volume_window: int = 20
    percentile_window: int = 63
    streak_percentile: float = 0.65

    def __post_init__(self):
        if self.volume_window < 2 or self.percentile_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.streak_percentile < 1.0:
            raise ValueError('streak_percentile must be between zero and one.')


class GeneratedStrategy(BaseStrategy[ImpactStreakParams]):
    strategy_id = 'gen_local-w2_1791324481'

    @classmethod
    def params_type(cls) -> type[ImpactStreakParams]:
        return ImpactStreakParams

    def warmup_bars(self, params: ImpactStreakParams) -> int:
        return int(params.volume_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: ImpactStreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].where(data['close'] > 0).pct_change(fill_method=None)
        volume = data['volume'].where(data['volume'] > 0)
        out['volume_median'] = volume.rolling(params.volume_window, min_periods=params.volume_window).median()
        out['relative_volume'] = volume / out['volume_median']
        out['impact'] = (out['return'] / out['relative_volume']).replace([np.inf, -np.inf], np.nan)
        out['impact_change'] = out['impact'].diff()
        out['direction'] = np.sign(out['impact_change']).fillna(0).astype(int)
        groups = out['direction'].ne(out['direction'].shift()).cumsum()
        out['streak'] = (out.groupby(groups).cumcount() + 1).astype(float)
        out.loc[out['direction'].eq(0), 'streak'] = 0.0
        out.loc[out['impact_change'].isna(), 'streak'] = np.nan
        out['streak_threshold'] = out['streak'].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.streak_percentile).shift(1)
        out['entry_direction'] = out['direction'].where(out['streak'].gt(out['streak_threshold']), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ImpactStreakParams) -> SignalFrame:
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, entry in enumerate(indicators['entry_direction'].to_numpy(dtype=int)):
            if position and entry == -position:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ImpactStreakParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_direction': [1, 0, 0, -1, -1, 0, 1, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    warmup = strategy.indicators(data, params)
    assert strategy.generate_signals(data, warmup, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
