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
    min_samples: int = 30
    area_threshold: float = 0.08
    season_score_threshold: float = 0.30
    atr_bars: int = 14
    stop_atr: float = 1.5
    holding_days: int = 2


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = 'gen_local-w2_1790985418'

    @classmethod
    def params_type(cls) -> type[SignatureSeasonParams]:
        return SignatureSeasonParams

    def warmup_bars(self, params: SignatureSeasonParams) -> int:
        return int(max(max(2, params.path_bars) + 1,
                       max(2, params.season_bars) + 1,
                       max(1, params.atr_bars) + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        opening = data['open'].astype(float)
        length = max(2, int(params.path_bars))
        season_length = max(2, int(params.season_bars))
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['intraday_return'] = (close / opening.where(opening > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(max(1, int(params.atr_bars)), min_periods=max(1, int(params.atr_bars))).mean()
        out['path_return'] = out['return'].rolling(length, min_periods=length).sum()
        out['path_variation'] = out['return'].abs().rolling(length, min_periods=length).sum()
        # Half the antisymmetric second-level signature for increments
        # (dt, dr) = (1 / length, daily return), oldest increment first.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            position = length - lag
            weight = (2.0 * position - length - 1.0) / (2.0 * length)
            area = area + weight * out['return'].shift(lag)
        out['path_area'] = area
        out['area_ratio'] = area / out['path_variation'].where(out['path_variation'] > 0)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        # Today's realized intraday return belongs to yesterday's decision weekday.
        origin_weekday = weekday.shift(1)
        out['season_mean'] = np.nan
        out['season_score'] = np.nan
        out['season_count'] = np.nan
        for day in range(5):
            observed = origin_weekday.eq(day) & out['intraday_return'].notna()
            sample = out['intraday_return'].where(observed, 0.0)
            count = observed.astype(float).rolling(season_length, min_periods=season_length).sum()
            total = sample.rolling(season_length, min_periods=season_length).sum()
            squares = sample.pow(2).rolling(season_length, min_periods=season_length).sum()
            mean = total / count.where(count > 0)
            variance = ((squares - total.pow(2) / count.where(count > 0)) / (count - 1).where(count > 1)).clip(lower=0.0)
            standard_error = np.sqrt(variance / count.where(count > 0))
            score = mean / standard_error.where(standard_error > 0)
            selected = weekday.eq(day)
            out.loc[selected, 'season_mean'] = mean.loc[selected]
            out.loc[selected, 'season_score'] = score.loc[selected]
            out.loc[selected, 'season_count'] = count.loc[selected]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        eligible = (
            indicators['area_ratio'].gt(params.area_threshold)
            & indicators['path_return'].lt(0.0)
            & indicators['season_mean'].gt(0.0)
            & indicators['season_score'].gt(params.season_score_threshold)
            & indicators['season_count'].ge(max(2, int(params.min_samples)))
            & indicators['atr'].gt(0.0)
        ).fillna(False)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        fixed_stop = np.nan
        age = 0
        horizon = min(2, max(1, int(params.holding_days)))
        stop_multiple = max(0.01, float(params.stop_atr))
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    age = 0
                    # Freeze the stop at the actual entry-bar open using
                    # ATR available when the preceding close authorized entry.
                    fixed_stop = opens[i] - stop_multiple * atr[i - 1]
            if active:
                age += 1
                stopped = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                if stopped or age >= horizon:
                    active = False
                    decisions[i] = 0
                else:
                    decisions[i] = 1
                continue
            if confirmed[i] and i + 1 < len(data):
                decisions[i] = 1
                pending_entry = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.bdate_range('2020-01-06', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 96.0,
                         'close': [100.0, 100.0, 100.0, 97.0, 100.0, 100.0],
                         'volume': 1000.0}, index=index)
    features = pd.DataFrame({'area_ratio': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
                             'path_return': -0.01, 'season_mean': 0.001,
                             'season_score': 1.0, 'season_count': 50.0,
                             'atr': 1.0}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = SignatureSeasonParams()
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    params.holding_days = 1
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
