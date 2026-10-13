from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    vol_window: int = 4
    percentile_window: int = 120
    shock_percentile: float = 0.90
    retention: float = 0.70
    atr_window: int = 14
    stop_atr: float = 1.50
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791272561'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(2 * params.vol_window + params.percentile_window + 3,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.vol_window < 2 or params.percentile_window < 2
                or params.atr_window < 1 or not 0 < params.shock_percentile < 1
                or not 0 < params.retention <= 1 or params.stop_atr <= 0
                or not 3 <= params.holding_bars <= 5):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['returns'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['returns'].rolling(params.vol_window).std(ddof=0)
        out['prior_volatility'] = out['volatility'].shift(params.vol_window)
        out['volatility_jump'] = out['volatility'] / out['prior_volatility'].replace(0, np.nan)
        out['volatility_jump'] = out['volatility_jump'].replace([np.inf, -np.inf], np.nan)
        out['threshold'] = out['volatility_jump'].shift(1).rolling(
            params.percentile_window
        ).quantile(params.shock_percentile)
        out['changepoint'] = (
            out['volatility_jump'].gt(out['threshold'])
            & out['volatility_jump'].shift(1).le(out['threshold'].shift(1))
        )
        out['event_anchor'] = close.shift(params.vol_window)
        out['event_displacement'] = close - out['event_anchor']
        out['retained_displacement'] = close - out['event_anchor'].shift(1)
        out['retention_ratio'] = (
            out['retained_displacement']
            / out['event_displacement'].shift(1).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        confirmed = (
            out['changepoint'].shift(1, fill_value=False)
            & out['retention_ratio'].ge(params.retention)
            & out['atr'].gt(0)
        )
        out['entry'] = np.sign(out['event_displacement'].shift(1)).where(
            confirmed, 0
        ).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        position = 0
        entry_price = np.nan
        stop_distance = np.nan
        held = 0
        for i in range(len(data)):
            if position:
                # The preceding close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                stopped = valid and position * (closes[i] - entry_price) < -stop_distance
                if not valid or stopped or held >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                    held = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
                held = 0
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[2], 'close'] = 97.0
    data.loc[index[6], 'close'] = 103.0
    indicators = pd.DataFrame({'atr': 2.0, 'entry': [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    params = StrategyParams(stop_atr=1.0)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    data['close'] = 100.0
    indicators['entry'] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
