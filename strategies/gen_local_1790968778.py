from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_window: int = 20
    pressure_window: int = 6
    rank_window: int = 126
    tail_percentile: float = 0.10
    profit_target: float = 0.04
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790968778'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.volume_window + params.pressure_window + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.volume_window, params.pressure_window, params.rank_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not 0.0 < params.tail_percentile < 0.5:
            raise ValueError('tail_percentile must be between zero and one half.')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.time_stop < 1:
            raise ValueError('Exit parameters must be positive.')
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(lambda s: s >= 0)
        location = (2.0 * data['close'] - data['high'] - data['low']) / out['range'].replace(0.0, np.nan)
        out['close_location'] = location.mask(out['range'].eq(0), 0.0).clip(-1.0, 1.0)
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0)
        out['capacity'] = out['valid_volume'].rolling(params.volume_window, min_periods=params.volume_window).mean().replace(0.0, np.nan)
        out['pressure'] = out['close_location'] * out['valid_volume'] / out['capacity']
        out['backlog'] = out['pressure'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['percentile'] = out['backlog'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['previous_percentile'] = out['percentile'].shift(1)
        out['backlog_change'] = out['backlog'].diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators[['backlog', 'previous_percentile', 'backlog_change']].notna().all(axis=1)
        long_entry = ready & indicators['previous_percentile'].le(params.tail_percentile) & indicators['backlog'].lt(0) & indicators['backlog_change'].gt(0)
        short_entry = ready & indicators['previous_percentile'].ge(1.0 - params.tail_percentile) & indicators['backlog'].gt(0) & indicators['backlog_change'].lt(0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                # ponytail: nominal open reference; broker fill accounting is needed for exact net-profit targets.
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else -np.inf
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.time_stop:
                    position = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[1], 'close'] = 103.0
    data.loc[index[4], 'close'] = 97.0
    indicators = pd.DataFrame({'backlog': 0.0, 'previous_percentile': 0.5, 'backlog_change': 0.0}, index=index)
    indicators.loc[index[[0, 6]], :] = [-2.0, 0.05, 0.2]
    indicators.loc[index[3], :] = [2.0, 0.95, -0.2]
    params = Params(profit_target=0.02, time_stop=3)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0]
    warmup = GeneratedStrategy.indicators(None, data, params)
    flat = GeneratedStrategy.generate_signals(None, data, warmup, None, params)
    assert flat.data['signal'].eq(0).all()
    assert result.data['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
