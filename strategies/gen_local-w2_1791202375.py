from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_window: int = 64
    frac_order: float = 0.4
    corr_window: int = 40
    percentile_window: int = 126
    entry_quantile: float = 0.8
    correlation_quantile: float = 0.7
    atr_window: int = 14
    hold_bars: int = 18
    breakeven_pct: float = 0.025
    trail_atr: float = 2.0
    initial_stop_atr: float = 3.0

    def __post_init__(self):
        periods = (self.frac_window, self.corr_window, self.percentile_window, self.atr_window, self.hold_bars)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 2 for x in periods):
            raise ValueError('Lookbacks and holding period must be integers >= 2')
        values = (self.frac_order, self.entry_quantile, self.correlation_quantile, self.breakeven_pct, self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(x) for x in values):
            raise ValueError('Parameters must be finite')
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError('frac_order must lie between zero and one')
        if not all(0.5 < x < 1.0 for x in (self.entry_quantile, self.correlation_quantile)):
            raise ValueError('Percentile gates must lie between 0.5 and one')
        if min(self.breakeven_pct, self.trail_atr, self.initial_stop_atr) <= 0:
            raise ValueError('Stop parameters must be positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791202375'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.frac_window + params.corr_window + params.percentile_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['log_close'] = np.log(close.where(close > 0))
        # ponytail: fixed-width fractional approximation; widen frac_window if tail accuracy matters.
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fractional = out['log_close'] * weights[0]
        for k in range(1, params.frac_window):
            fractional = fractional + weights[k] * out['log_close'].shift(k)
        out['fractional_close'] = fractional
        out['innovation'] = fractional.diff()
        out['lagged_innovation'] = out['innovation'].shift(1)
        out['correlation'] = out['innovation'].rolling(params.corr_window).corr(out['lagged_innovation']).clip(-1.0, 1.0)
        out['innovation_scale'] = out['innovation'].rolling(params.corr_window).std(ddof=0).replace(0.0, np.nan)
        out['forecast'] = out['correlation'] * out['innovation'] / out['innovation_scale']
        out['forecast_magnitude'] = out['forecast'].abs()
        out['correlation_strength'] = out['correlation'].abs()
        out['forecast_threshold'] = out['forecast_magnitude'].shift(1).rolling(params.percentile_window).quantile(params.entry_quantile)
        out['correlation_threshold'] = out['correlation_strength'].shift(1).rolling(params.percentile_window).quantile(params.correlation_quantile)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['forecast', 'forecast_threshold', 'correlation_strength', 'correlation_threshold', 'atr']].notna().all(axis=1)
        eligible = ready & (out['atr'] > 0) & (out['forecast_magnitude'] > out['forecast_threshold']) & (out['correlation_strength'] > out['correlation_threshold'])
        out['candidate'] = np.sign(out['forecast'].fillna(0.0)).where(eligible, 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        position = 0
        entry = 0.0
        floor = -np.inf
        peak = -np.inf
        armed = False
        held = 0
        for i in range(n):
            # Yesterday's decision fills at today's open; no future open is read.
            if position == 0 and i > 0 and raw[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                position = int(raw[i - 1])
                entry = float(opens[i])
                floor = position * entry - params.initial_stop_atr * atr[i - 1]
                peak = position * entry
                armed = False
                held = 0
            if position:
                held += 1
                if held >= params.hold_bars:
                    position = 0
                    continue
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    raw[i] = position
                    continue
                q = position * closes[i]
                # Close-observed stops exit at the next open through the final shift.
                if q <= floor:
                    position = 0
                    continue
                peak = max(peak, q)
                if q - position * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    floor = max(floor, position * entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        # Direction-adjusted floor only rises; a short's price stop falls.
                        floor = max(floor, peak - params.trail_atr * atr[i])
                raw[i] = position
            else:
                raw[i] = int(candidates[i])
        df = pd.DataFrame({'signal': raw, 'size': np.ones(n, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    params = Params()
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0.0, 0.0, 3.0, 4.0, 0.0, 0.0])
        data = pd.DataFrame({'open': prices, 'high': prices + 1.0, 'low': prices - 1.0, 'close': prices, 'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({'atr': 1.0, 'candidate': [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert np.array_equal(result['signal'].to_numpy(), direction * np.array([0, 1, 1, 1, 1, 0]))
        assert (result['size'] > 0).all()
    data.loc[:, ['open', 'high', 'low', 'close']] = 100.0
    indicators['candidate'] = [1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, None, Params(hold_bars=3)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['candidate'].eq(0).all()


if __name__ == '__main__':
    _self_check()
