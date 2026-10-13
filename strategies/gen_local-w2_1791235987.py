from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 126
    archive_bars: int = 504
    analog_radius: float = 0.18
    min_analogs: int = 12
    entry_agreement: float = 0.40
    holding_bars: int = 18
    atr_window: int = 14
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791235987'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(params.atr_window + 1, params.rank_window + 12 + params.holding_bars + params.archive_bars))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.rank_window < 2 or params.archive_bars < 1
                or params.min_analogs < 1 or params.min_analogs > params.archive_bars
                or params.holding_bars < 1 or params.atr_window < 1
                or not np.isfinite(params.analog_radius) or not 0 < params.analog_radius <= 1
                or not np.isfinite(params.entry_agreement) or not 0 < params.entry_agreement < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError('Invalid analog parameters')
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        out['return4'] = close.pct_change(4, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['rank'] = out['return4'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['early_rank'] = out['rank'].shift(8)
        out['middle_rank'] = out['rank'].shift(4)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['atr_fraction'] = out['atr'] / close.where(close > 0)
        features = out[['early_rank', 'middle_rank', 'rank']].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        volatility = out['atr_fraction'].to_numpy(dtype=float)
        agreement = np.zeros(len(data), dtype=float)
        counts = np.zeros(len(data), dtype=int)
        first = params.rank_window + 11
        horizon = params.holding_bars
        # ponytail: scan a bounded archive per bar; index analogs if intraday scale requires it.
        for i in range(GeneratedStrategy.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Both the analog's outcome and its path end before the current 12-bar shape begins.
            end = i - horizon - 12 + 1
            start = max(first, end - params.archive_bars)
            candidates = np.arange(start, end, dtype=int)
            if len(candidates) == 0:
                continue
            distance = np.max(np.abs(features[candidates] - features[i]), axis=1)
            valid = (np.isfinite(distance) & (distance <= params.analog_radius)
                     & np.isfinite(prices[candidates]) & (prices[candidates] > 0)
                     & np.isfinite(prices[candidates + horizon])
                     & np.isfinite(volatility[candidates]) & (volatility[candidates] > 0))
            matches = candidates[valid]
            counts[i] = len(matches)
            if len(matches) < params.min_analogs:
                continue
            outcomes = (prices[matches + horizon] / prices[matches] - 1.0)
            outcomes = np.clip(outcomes / (volatility[matches] * np.sqrt(horizon)), -3.0, 3.0)
            magnitude = np.abs(outcomes).sum()
            if magnitude > 0:
                agreement[i] = float(outcomes.sum() / magnitude)
        out['analog_count'] = counts
        out['agreement'] = agreement
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        agreement = indicators['agreement'].fillna(0.0).to_numpy(dtype=float)
        counts = indicators['analog_count'].fillna(0).to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        high_water = np.nan
        trailing_floor = np.nan
        short_ceiling = np.nan
        entry_size = 0.20
        ready = GeneratedStrategy.warmup_bars(params) - 1
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            exited = False
            if position:
                expired = i - entry_bar >= params.holding_bars
                stopped = False
                if valid:
                    high_water = max(high_water, close[i])
                    trailing_floor = max(trailing_floor, high_water - params.trail_atr * atr[i])
                    # The required highest-close trailing exit applies to either trade direction.
                    stopped = close[i] <= trailing_floor
                    # Also bound adverse short moves with an entry-ATR ceiling.
                    if position == -1:
                        stopped = stopped or close[i] >= short_ceiling
                if expired or stopped or not valid:
                    position = 0
                    exited = True
            if (position == 0 and not exited and i >= ready and valid
                    and counts[i] >= params.min_analogs
                    and np.isfinite(agreement[i])
                    and abs(agreement[i]) >= params.entry_agreement):
                position = 1 if agreement[i] > 0 else -1
                entry_bar = i
                high_water = close[i]
                trailing_floor = close[i] - params.trail_atr * atr[i]
                short_ceiling = close[i] + params.trail_atr * atr[i]
                entry_size = float(0.20 + 0.75 * np.clip(abs(agreement[i]), 0.0, 1.0))
            signals[i] = position
            sizes[i] = entry_size if position else 0.20
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AnalogParams(rank_window=20, archive_bars=20, min_analogs=3)
    index = pd.date_range('2020-01-01', periods=160, freq='B')
    t = np.arange(160, dtype=float)
    price = 100.0 + 0.02 * t + 2.0 * np.sin(t / 5.0)
    data = pd.DataFrame({'open': price, 'high': price + 1.0,
                         'low': price - 1.0, 'close': price,
                         'volume': np.full(160, 1000.0)}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:140], params)
    assert full.iloc[:140].equals(prefix), 'Analog forecast must be causal'
    left = GeneratedStrategy.generate_signals(data, full, None, params).data
    right = GeneratedStrategy.generate_signals(data.iloc[:140], prefix, None, params).data
    assert left.iloc[:140].equals(right), 'Signals and sizes must be causal'
    forced = pd.DataFrame({'atr': 1.0, 'agreement': 0.8,
                           'analog_count': 20}, index=index)
    flat = data.copy()
    flat['close'] = 100.0
    flat.iloc[80, flat.columns.get_loc('close')] = 110.0
    flat.iloc[81, flat.columns.get_loc('close')] = 106.0
    result = GeneratedStrategy.generate_signals(flat, forced, None, params).data
    assert result['signal'].iloc[69] == 0 and result['signal'].iloc[70] == 1
    assert result['signal'].iloc[81] == 1 and result['signal'].iloc[82] == 0
    assert result['signal'].isin([-1, 0, 1]).all()
    assert np.isfinite(result['size']).all() and (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
