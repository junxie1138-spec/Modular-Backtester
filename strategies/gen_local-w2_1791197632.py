from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 5
    season_bars: int = 252
    area_threshold: float = 0.08
    min_season_return: float = 0.0001
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = 'gen_local-w2_1791197632'

    @classmethod
    def params_type(cls) -> type[SignatureSeasonParams]:
        return SignatureSeasonParams

    @staticmethod
    def warmup_bars(params: SignatureSeasonParams) -> int:
        if params.path_bars < 2 or params.season_bars < 45:
            raise ValueError('path_bars must be >= 2 and season_bars >= 45')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if not np.isfinite(params.area_threshold) or params.area_threshold <= 0:
            raise ValueError('area_threshold must be finite and positive')
        if not np.isfinite(params.min_season_return):
            raise ValueError('min_season_return must be finite')
        return int(max(params.path_bars, params.season_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['return'] = np.log(data['close'].where(data['close'] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_change'] = np.log(data['volume'].where(data['volume'] > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r = out['return']
        v = out['volume_change']
        length = params.path_bars
        area = pd.Series(0.0, index=data.index)
        # Second-order antisymmetric signature: half the sum over all ordered increment pairs.
        for lag in range(1, length):
            cross = r.shift(lag) * v - v.shift(lag) * r
            area = area + cross.rolling(length - lag, min_periods=length - lag).sum()
        out['signature_area'] = 0.5 * area
        out['path_scale'] = r.abs().rolling(length, min_periods=length).sum() * v.abs().rolling(length, min_periods=length).sum()
        out['normalized_area'] = (out['signature_area'] / out['path_scale'].where(out['path_scale'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['weekday'] = data.index.dayofweek
        out['previous_weekday'] = out['weekday'].shift(1)
        seasonal_edge = pd.Series(np.nan, index=data.index, dtype=float)
        minimum = max(3, params.season_bars // 15)
        for weekday in range(5):
            # Today's realized return trains yesterday's decision weekday, using no future returns.
            historical = r.where(out['previous_weekday'].eq(weekday))
            estimate = historical.rolling(params.season_bars, min_periods=minimum).mean()
            seasonal_edge = seasonal_edge.where(~out['weekday'].eq(weekday), estimate)
        out['seasonal_edge'] = seasonal_edge
        out['ready'] = np.arange(len(data)) >= warmup - 1
        out['setup'] = (
            out['ready']
            & out['normalized_area'].lt(-params.area_threshold)
            & out['seasonal_edge'].gt(params.min_season_return)
            & out['return'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        self.warmup_bars(params)
        setup = indicators['setup'].fillna(False).astype(bool)
        confirmed = setup & setup.shift(1, fill_value=False)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(confirmed.to_numpy(dtype=bool)):
            if remaining > 0:
                remaining -= 1
                if remaining > 0:
                    signal[i] = 1
            elif enter:
                remaining = params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'setup': [False, True, True, True, True, False, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, SignatureSeasonParams())
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert result.data['size'].gt(0).all()
