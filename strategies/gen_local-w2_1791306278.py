from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SemivolatilityParams:
    lookback: int = 15
    contraction_lag: int = 5
    dominance: float = 0.65
    minimum_snr: float = 1.1
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[SemivolatilityParams]):
    strategy_id = 'gen_local-w2_1791306278'

    @classmethod
    def params_type(cls) -> type[SemivolatilityParams]:
        return SemivolatilityParams

    def warmup_bars(self, params: SemivolatilityParams) -> int:
        return int(max(200, params.lookback + params.contraction_lag + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SemivolatilityParams) -> pd.DataFrame:
        close = data['close']
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upside = returns.clip(lower=0).pow(2).rolling(params.lookback, min_periods=params.lookback).mean()
        downside = returns.clip(upper=0).pow(2).rolling(params.lookback, min_periods=params.lookback).mean()
        variance = upside + downside
        share = upside.div(variance.where(variance > 0))
        energy = returns.pow(2).rolling(params.lookback, min_periods=params.lookback).sum()
        drift = returns.rolling(params.lookback, min_periods=params.lookback).sum()
        snr = drift.div(np.sqrt(energy.where(energy > 0)))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out['return'] = returns
        out['upside_semivariance'] = upside
        out['downside_semivariance'] = downside
        out['lagged_downside_semivariance'] = downside.shift(params.contraction_lag)
        out['upside_share'] = share
        out['snr'] = snr
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['entry'] = (
            (share > params.dominance)
            & (share.shift(1) <= params.dominance)
            & (downside < out['lagged_downside_semivariance'])
            & (snr >= params.minimum_snr)
            & (close > out['ma200'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SemivolatilityParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        desired = np.zeros(len(data), dtype=int)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        high = data['high'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        candidate = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = stop = peak = 0.0
        held = 0
        for i in range(len(data)):
            # Yesterday's target enters at today's open, before today's close decision.
            if not active and i > 0 and desired[i - 1] == 1:
                if np.isfinite(opening[i]) and opening[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry = opening[i]
                    peak = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    held = 0
            if active:
                held += 1
                valid = np.isfinite(close[i]) and np.isfinite(high[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i])
                if not valid:
                    active = False
                    continue
                peak = max(peak, high[i])
                if high[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Stops are evaluated at the close; exits fill at the next open.
                if close[i] <= stop or close[i] <= regime[i] or held >= params.max_hold:
                    active = False
                    continue
                desired[i] = 1
            elif candidate[i]:
                desired[i] = 1
        df['signal'] = desired
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=205, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[200], ['high', 'close']] = 110.0
    data.loc[index[201], ['high', 'close']] = 103.0
    indicators = pd.DataFrame({'atr': 1.0, 'ma200': 90.0, 'entry': False}, index=index)
    indicators.loc[index[200], 'entry'] = True
    indicators.loc[index[202], 'atr'] = 10.0
    strategy = GeneratedStrategy()
    params = SemivolatilityParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[200:204].tolist() == [0, 1, 1, 0]
    assert result['signal'].isin([0, 1]).all()
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= 200
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert not calculated['entry'].iloc[:199].any()


if __name__ == '__main__':
    _self_check()
