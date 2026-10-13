from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fast_window: int = 5
    slow_window: int = 20
    snapshot_gap: int = 4
    history_bars: int = 252
    hold_bars: int = 18
    shape_tolerance: float = 0.18
    min_matches: int = 8
    min_edge: float = 0.015
    recovery_window: int = 5
    efficiency_threshold: float = 0.25

    def __post_init__(self):
        windows = (self.fast_window, self.slow_window, self.snapshot_gap,
                   self.history_bars, self.hold_bars, self.min_matches,
                   self.recovery_window)
        if any(type(value) is not int or value < 1 for value in windows):
            raise ValueError('Window parameters must be positive integers.')
        if not 2 <= self.fast_window < self.slow_window:
            raise ValueError('Require 2 <= fast_window < slow_window.')
        if self.min_matches > self.history_bars:
            raise ValueError('min_matches cannot exceed history_bars.')
        thresholds = (self.shape_tolerance, self.min_edge,
                      self.efficiency_threshold)
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError('Thresholds must be finite.')
        if self.shape_tolerance <= 0 or self.min_edge < 0:
            raise ValueError('Require positive tolerance and nonnegative edge.')
        if not 0 < self.efficiency_threshold < 1:
            raise ValueError('efficiency_threshold must lie between zero and one.')


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = 'gen_local-w2_1791326494'

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(
            params.slow_window + 2 * params.snapshot_gap
            + params.hold_bars + params.history_bars + 2,
            2 * params.recovery_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out['fast_vol'] = out['return'].rolling(
            params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out['slow_vol'] = out['return'].rolling(
            params.slow_window, min_periods=params.slow_window).std(ddof=0)
        ratio = out['fast_vol'] / out['slow_vol'].where(out['slow_vol'] > 0)
        out['wave_now'] = np.log(ratio.where(ratio > 0))
        out['wave_mid'] = out['wave_now'].shift(params.snapshot_gap)
        out['wave_old'] = out['wave_now'].shift(2 * params.snapshot_gap)
        wave = out[['wave_now', 'wave_mid', 'wave_old']]
        valid_wave = pd.Series(
            np.isfinite(wave.to_numpy()).all(axis=1), index=data.index)
        matches = pd.Series(0, index=data.index, dtype=int)
        outcome_sum = pd.Series(0.0, index=data.index)
        # ponytail: scan a bounded history; use chunked arrays if history grows substantially.
        for lag in range(params.hold_bars + 2,
                         params.hold_bars + 2 + params.history_bars):
            historical = wave.shift(lag)
            distance = (wave - historical).abs().max(axis=1, skipna=False)
            # Analog entry is next open; exit is exactly hold_bars opens later.
            # The exit open is at least one completed bar behind this decision.
            outcome = (opening.shift(lag - params.hold_bars - 1)
                       / opening.shift(lag - 1) - 1.0)
            match = (valid_wave & distance.le(params.shape_tolerance)
                     & pd.Series(np.isfinite(outcome.to_numpy()), index=data.index))
            matches += match.astype(int)
            outcome_sum += outcome.where(match, 0.0)
        out['analog_matches'] = matches
        out['analog_edge'] = outcome_sum / matches.where(matches > 0)
        out['price_change'] = close.diff()
        out['price_travel'] = out['price_change'].abs().rolling(
            params.recovery_window, min_periods=params.recovery_window).sum()
        out['efficiency'] = (close.diff(params.recovery_window)
                             / out['price_travel'].where(out['price_travel'] > 0))
        out['prior_efficiency'] = out['efficiency'].shift(params.recovery_window)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        volatility_agrees = (
            indicators['analog_matches'].ge(params.min_matches)
            & indicators['analog_edge'].gt(params.min_edge)
        )
        price_agrees = (
            indicators['efficiency'].gt(params.efficiency_threshold)
            & indicators['prior_efficiency'].lt(-params.efficiency_threshold)
        )
        entries = (volatility_agrees & price_agrees).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entered = -1
        for i in range(len(data)):
            if entered >= 0:
                if i - entered == params.hold_bars:
                    entered = -1
                    continue
                signal[i] = 1
            elif entries[i]:
                entered = i
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = WaveParams()
    index = pd.date_range('2020-01-01', periods=2 * params.hold_bars + 4)
    data = pd.DataFrame(100.0, index=index,
                        columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = pd.DataFrame({
        'analog_matches': 20,
        'analog_edge': 0.03,
        'efficiency': 0.5,
        'prior_efficiency': -0.5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:params.hold_bars + 1].eq(1).all()
    assert result['signal'].iloc[params.hold_bars + 1] == 0
    assert result['signal'].iloc[params.hold_bars + 2:2 * params.hold_bars + 2].eq(1).all()
    assert result['signal'].iloc[2 * params.hold_bars + 2] == 0
    indicators['prior_efficiency'] = np.nan
    flat = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert flat['signal'].eq(0).all()
