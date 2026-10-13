from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnchorParams:
    anchor_bars: int = 20
    excursion_bars: int = 2
    min_depth: float = 0.04
    volume_multiple: float = 1.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[AnchorParams]):
    strategy_id = 'gen_local_1791216263'

    @classmethod
    def params_type(cls) -> type[AnchorParams]:
        return AnchorParams

    def warmup_bars(self, params: AnchorParams) -> int:
        return int(max(200, params.anchor_bars + params.excursion_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AnchorParams) -> pd.DataFrame:
        if params.anchor_bars < 2 or params.excursion_bars < 1:
            raise ValueError('Lookbacks must be positive; anchor_bars must be at least two.')
        if not np.isfinite(params.min_depth) or params.min_depth <= 0:
            raise ValueError('min_depth must be positive and finite.')
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 1:
            raise ValueError('volume_multiple must exceed one and be finite.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        volume = data['volume'].where(data['volume'] > 0)
        length = params.anchor_bars
        out['price_volume'] = close * volume
        out['prior_volume_sum'] = volume.rolling(length, min_periods=length).sum().shift(1)
        out['prior_price_volume_sum'] = out['price_volume'].rolling(length, min_periods=length).sum().shift(1)
        out['anchor'] = out['prior_price_volume_sum'] / out['prior_volume_sum'].replace(0, np.nan)
        out['prior_high'] = data['high'].rolling(length, min_periods=length).max().shift(1)
        out['prior_low'] = data['low'].rolling(length, min_periods=length).min().shift(1)
        out['prior_range'] = (out['prior_high'] - out['prior_low']).where(out['prior_high'] > out['prior_low'])
        out['relative_position'] = (close - out['anchor']) / out['prior_range']
        out['prior_volume_mean'] = volume.rolling(length, min_periods=length).mean().shift(1)
        out['volume_ratio'] = volume / out['prior_volume_mean'].replace(0, np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        residence = params.excursion_bars
        out['excursion_max'] = out['relative_position'].shift(1).rolling(residence, min_periods=residence).max()
        out['excursion_min'] = out['relative_position'].shift(1).rolling(residence, min_periods=residence).min()
        out['quiet_volume_max'] = out['volume_ratio'].shift(1).rolling(residence, min_periods=residence).max()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['quiet_excursion'] = out['quiet_volume_max'].lt(1.0)
        out['volume_confirmed'] = out['volume_ratio'].ge(params.volume_multiple)
        out['long_entry'] = (
            out['quiet_excursion'] & out['volume_confirmed']
            & out['excursion_max'].lt(-params.min_depth)
            & out['relative_position'].gt(0)
            & close.gt(out['ma200'])
        ).fillna(False)
        out['short_entry'] = (
            out['quiet_excursion'] & out['volume_confirmed']
            & out['excursion_min'].gt(params.min_depth)
            & out['relative_position'].lt(0)
            & close.lt(out['ma200'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnchorParams) -> SignalFrame:
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('Holding period and profit target must be positive.')
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    entry_price = np.nan
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    target_hit = np.isfinite(gain) and gain >= params.profit_target
                    time_up = i - entry_bar + 1 >= params.max_hold_bars
                    if target_hit or time_up:
                        position = 0
                        entry_price = np.nan
            elif np.isfinite(closes[i]) and closes[i] > 0:
                if longs[i]:
                    position = 1
                    entry_price = np.nan
                elif shorts[i]:
                    position = -1
                    entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=13, freq='B')
    sample = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    sample.iloc[2, sample.columns.get_loc('close')] = 103.0
    sample.iloc[6, sample.columns.get_loc('close')] = 97.0
    features = pd.DataFrame({'long_entry': False, 'short_entry': False}, index=index)
    features.iloc[[0, 8], features.columns.get_loc('long_entry')] = True
    features.iloc[4, features.columns.get_loc('short_entry')] = True
    params = AnchorParams(profit_target=0.02, max_hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, sample, features, None, params).data
    expected = [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result['signal'].tolist() == expected
    assert result['size'].gt(0).all()
    for length in range(1, len(sample) + 1):
        prefix = GeneratedStrategy.generate_signals(None, sample.iloc[:length], features.iloc[:length], None, params).data
        assert prefix['signal'].tolist() == expected[:length]
