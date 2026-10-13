from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    history: int = 252
    neighbors: int = 12
    forecast_floor: float = 0.001
    pressure_floor: float = 0.15
    profit_target: float = 0.008
    max_hold: int = 2

    def __post_init__(self):
        if self.history < 40 or not 1 <= self.neighbors <= self.history - 10:
            raise ValueError('Invalid analog sample sizes')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be 1 or 2')
        values = [self.forecast_floor, self.pressure_floor, self.profit_target]
        if not np.isfinite(values).all():
            raise ValueError('Thresholds must be finite')
        if self.forecast_floor < 0 or not 0 <= self.pressure_floor < 1 or self.profit_target <= 0:
            raise ValueError('Invalid thresholds')


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = 'gen_local_1791276930'

    @classmethod
    def params_type(cls):
        return GapAnalogParams

    def warmup_bars(self, params: GapAnalogParams) -> int:
        return int(params.history + 26 + params.max_hold)

    def indicators(self, data: pd.DataFrame, params: GapAnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['gap'] = opening / close.shift(1) - 1.0
        out['gap_scale'] = out['gap'].rolling(20, min_periods=20).std().shift(1)
        out['gap_z'] = out['gap'] / out['gap_scale'].where(out['gap_scale'] > 1e-8)
        out['gap_z_lag1'] = out['gap_z'].shift(1)
        out['gap_z_lag2'] = out['gap_z'].shift(2)
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_pressure'] = ((2 * close - data['high'] - data['low']) / out['range']).clip(-1, 1)
        volume = data['volume'].where(data['volume'] > 0)
        out['signed_volume'] = out['close_pressure'] * volume
        out['pressure'] = out['signed_volume'].rolling(3, min_periods=3).sum() / volume.rolling(3, min_periods=3).sum()
        # At bar t this label belongs to the decision at t - max_hold.
        out['matured_return'] = close / opening.shift(params.max_hold - 1) - 1.0
        shape = out[['gap_z', 'gap_z_lag1', 'gap_z_lag2']].to_numpy()
        labels = out['matured_return'].to_numpy()
        forecasts = np.full(len(data), np.nan)
        wins = np.full(len(data), np.nan)
        # ponytail: bounded O(bars * history) search; index only if intraday scale demands it.
        for i in range(self.warmup_bars(params), len(data)):
            if not np.isfinite(shape[i]).all():
                continue
            # Embargo keeps candidate features and outcomes outside the current three-gap shape.
            candidates = np.arange(max(23, i - params.history), i - params.max_hold - 2)
            outcomes = labels[candidates + params.max_hold]
            valid = np.isfinite(shape[candidates]).all(axis=1) & np.isfinite(outcomes)
            candidates = candidates[valid]
            outcomes = outcomes[valid]
            if len(candidates) < params.neighbors:
                continue
            distance = np.square(shape[candidates] - shape[i]).sum(axis=1)
            nearest = np.argsort(distance, kind='stable')[:params.neighbors]
            forecasts[i] = outcomes[nearest].mean()
            wins[i] = (outcomes[nearest] > 0).mean()
        out['analog_mean'] = forecasts
        out['analog_win_rate'] = wins
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAnalogParams,
    ) -> SignalFrame:
        gap_vote = (
            indicators['gap'].lt(0)
            & indicators['analog_mean'].gt(params.forecast_floor)
            & indicators['analog_win_rate'].ge(0.60)
        )
        pressure_vote = indicators['pressure'].gt(params.pressure_floor)
        entries = (gap_vote & pressure_vote).fillna(False).to_numpy(dtype=bool)
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opening[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_price and np.isfinite(closing[i]) and closing[i] >= entry_price * (1 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.max_hold
                # Close-observed exits execute at the following open, like entries.
                if target_hit or time_stop or not valid_price:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(closing[i]) and closing[i] > 0:
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapAnalogParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'gap': -0.01, 'analog_mean': 0.01, 'analog_win_rate': 0.8, 'pressure': 0.5}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert signals['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    data.loc[index[1], 'close'] = 102.0
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].tolist() == [0, 1, 0, 1, 1, 0]
    ind['pressure'] = np.nan
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].eq(0).all()
    assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
