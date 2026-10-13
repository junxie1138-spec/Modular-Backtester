from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 10
    history_bars: int = 252
    neighbors: int = 3
    percentile_bars: int = 126
    entry_percentile: float = 0.75
    holding_bars: int = 18
    atr_bars: int = 14
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791339346'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(params.atr_bars + 1,
                       params.history_bars + params.shape_bars
                       + params.percentile_bars + 2))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.shape_bars < 3 or params.holding_bars < 1
                or params.neighbors < 1 or params.percentile_bars < 2
                or params.atr_bars < 1 or params.trail_atr <= 0
                or not 0 < params.entry_percentile < 1):
            raise ValueError('Invalid analog strategy parameters')
        spacing = params.shape_bars + params.holding_bars
        lags = list(range(spacing, params.history_bars + 1, spacing))
        if len(lags) < params.neighbors:
            raise ValueError('History must contain enough mature analogs')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        correlations = []
        outcomes = []
        for lag in lags:
            correlation_name = f'correlation_{lag}'
            outcome_name = f'outcome_{lag}'
            out[correlation_name] = out['return'].rolling(params.shape_bars).corr(
                out['return'].shift(lag)).clip(-1.0, 1.0)
            # Both the analog and its subsequent outcome have already finished.
            out[outcome_name] = close.shift(lag - params.holding_bars) / close.shift(lag) - 1.0
            correlations.append(correlation_name)
            outcomes.append(outcome_name)
        similarities = out[correlations].to_numpy(dtype=float)
        realized = out[outcomes].to_numpy(dtype=float)
        valid = np.isfinite(similarities) & np.isfinite(realized)
        scores = np.where(valid, similarities, -np.inf)
        selected = np.argsort(-scores, axis=1, kind='stable')[:, :params.neighbors]
        matched = np.take_along_axis(realized, selected, axis=1)
        complete = valid.all(axis=1)
        matched = np.where(np.isfinite(matched), matched, 0.0)
        out['forecast'] = np.where(complete, np.median(matched, axis=1), np.nan)
        out['entry_threshold'] = out['forecast'].rolling(
            params.percentile_bars).quantile(params.entry_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        entries = ((indicators['forecast'] > indicators['entry_threshold'])
                   & (indicators['forecast'] > 0)).fillna(False).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if not (np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(opens[i]) and opens[i] > 0
                    and np.isfinite(atrs[i]) and atrs[i] > 0):
                active = False
                continue
            if active:
                age += 1
                if not np.isfinite(high_water):
                    # The previous close's decision fills at today's open.
                    high_water = closes[i]
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.holding_bars:
                    active = False
                else:
                    signals[i] = 1
            elif i + 1 >= warmup and entries[i]:
                active = True
                high_water = np.nan
                stop = -np.inf
                age = 0
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AnalogParams()
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, 700)))
    data = pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close, 'volume': 1000000},
                        index=pd.date_range('2020-01-01', periods=len(close)))
    full = GeneratedStrategy.indicators(None, data, params)
    prefix = GeneratedStrategy.indicators(None, data.iloc[:550], params)
    pd.testing.assert_frame_equal(full.iloc[:550], prefix)
    signals = GeneratedStrategy.generate_signals(None, data, full, None, params).data
    short = GeneratedStrategy.generate_signals(None, data.iloc[:550], prefix, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:550], short)
    assert signals['signal'].isin([0, 1]).all() and (signals['size'] > 0).all()
    assert signals['signal'].iloc[0] == 0
    forced = full.copy()
    forced['forecast'] = 1.0
    forced['entry_threshold'] = 0.5
    forced['atr'] = 1.0
    path = data.copy()
    path[['open', 'close']] = 100.0
    start = GeneratedStrategy.warmup_bars(params) - 1
    path.iloc[start + 2, path.columns.get_loc('close')] = 110.0
    path.iloc[start + 3, path.columns.get_loc('close')] = 106.0
    result = GeneratedStrategy.generate_signals(None, path, forced, None, params).data
    assert result['signal'].iloc[start] == 0
    assert result['signal'].iloc[start + 1] == 1
    assert result['signal'].iloc[start + 3] == 1
    assert result['signal'].iloc[start + 4] == 0


if __name__ == '__main__':
    _self_check()
