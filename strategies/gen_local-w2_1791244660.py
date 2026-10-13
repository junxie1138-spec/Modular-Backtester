from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    innovation_threshold: float = 0.5
    holding_bars: int = 4

    def __post_init__(self):
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold < 0:
            raise ValueError('innovation_threshold must be finite and nonnegative')
        if not isinstance(self.holding_bars, int) or not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be an integer from 3 through 5')


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = 'gen_local-w2_1791244660'

    @classmethod
    def params_type(cls) -> type[SeasonalSlopeParams]:
        return SeasonalSlopeParams

    def warmup_bars(self, params: SeasonalSlopeParams) -> int:
        return 137

    def indicators(self, data: pd.DataFrame, params: SeasonalSlopeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close.gt(0)))
        out['latent_level'] = out['log_close'].rolling(5, min_periods=5).mean()
        out['latent_slope'] = (out['latent_level'] - out['latent_level'].shift(5)) / 5.0
        out['innovation'] = out['latent_slope'].diff()
        out['past_innovation'] = out['innovation'].shift(1)
        out['innovation_scale'] = out['past_innovation'].rolling(126, min_periods=126).std(ddof=0).replace(0.0, np.nan)
        out['weekday'] = data.index.dayofweek
        out['weekday_mean'] = np.nan
        for day in range(5):
            selected = out['innovation'].where(out['weekday'].eq(day)).shift(1)
            out[f'weekday_{day}_past'] = selected
            out[f'weekday_{day}_sum'] = selected.fillna(0.0).rolling(126, min_periods=126).sum()
            out[f'weekday_{day}_count'] = selected.notna().astype(float).rolling(126, min_periods=126).sum()
            count = out[f'weekday_{day}_count']
            out[f'weekday_{day}_mean'] = out[f'weekday_{day}_sum'] / count.where(count.ge(15))
            mask = out['weekday'].eq(day)
            out.loc[mask, 'weekday_mean'] = out.loc[mask, f'weekday_{day}_mean']
        out['innovation_z'] = (out['innovation'] - out['weekday_mean']) / out['innovation_scale']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalSlopeParams) -> SignalFrame:
        eligible = (indicators['weekday_mean'].gt(0.0) & indicators['innovation_z'].gt(params.innovation_threshold)).fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(eligible):
            if exit_at >= 0:
                if i < exit_at:
                    target[i] = 1
                else:
                    # Force the timed exit even if another entry is eligible.
                    exit_at = -1
            elif enter:
                target[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({'signal': target, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.bdate_range('2020-01-01', periods=18)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'weekday_mean': 0.01, 'innovation_z': 2.0}, index=index)
    for holding_bars in (3, 4, 5):
        params = SeasonalSlopeParams(holding_bars=holding_bars)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        expected = [0] + [int(i % (holding_bars + 1) < holding_bars) for i in range(17)]
        assert result['signal'].tolist() == expected
        assert result['size'].gt(0).all()
    indicators.loc[:, 'innovation_z'] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SeasonalSlopeParams()).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
