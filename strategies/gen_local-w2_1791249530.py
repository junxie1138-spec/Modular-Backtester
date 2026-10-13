from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    memory_window: int = 60
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local-w2_1791249530'

    @classmethod
    def params_type(cls):
        return EchoParams

    @staticmethod
    def warmup_bars(params: EchoParams) -> int:
        return int(params.memory_window) + 73

    @staticmethod
    def indicators(data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.memory_window < 10 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('memory_window must be >= 10 and trail_atr must be positive')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        # Fixed 48-term fractional difference with order 0.4.
        weights = np.ones(48, dtype=float)
        for j in range(1, 48):
            weights[j] = -weights[j - 1] * (0.4 - j + 1) / j
        fractional = close.copy()
        for j in range(1, 48):
            fractional = fractional + weights[j] * close.shift(j)
        out['fractional_close'] = fractional
        out['innovation'] = fractional.diff()
        out['innovation_mean'] = out['innovation'].rolling(20, min_periods=20).mean()
        out['innovation_std'] = out['innovation'].rolling(20, min_periods=20).std(ddof=0)
        out['standardized'] = (out['innovation'] - out['innovation_mean']) / out['innovation_std'].replace(0.0, np.nan)
        out['lag_one'] = out['innovation'].shift(1)
        out['lag_five'] = out['innovation'].shift(5)
        window = int(params.memory_window)
        out['rho_one'] = out['innovation'].rolling(window, min_periods=window).corr(out['lag_one'])
        out['rho_five'] = out['innovation'].rolling(window, min_periods=window).corr(out['lag_five'])
        out['echo_source'] = out['standardized'].shift(4)
        out['forecast'] = out['rho_five'] * out['echo_source']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        eligible = (
            (out['rho_five'] > 0.20)
            & (out['rho_five'] > out['rho_one'].abs() + 0.10)
            & (out['forecast'].abs() > 0.35)
            & (out['atr'] > 0)
        )
        out['candidate'] = np.sign(out['forecast']).where(eligible, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        position = 0
        entry = np.nan
        entry_bar = -1
        stop = -np.inf
        peak = -np.inf
        armed = False
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry = opens[i]
                    entry_bar = i
                    peak = position * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                price = closes[i]
                if not np.isfinite(price) or price <= 0:
                    raw[i] = position
                    continue
                signed_price = position * price
                # Signed-price stops only increase; short price stops only decrease.
                exited = signed_price <= stop or i - entry_bar + 1 >= 10
                if not exited:
                    peak = max(peak, signed_price)
                    if peak - position * entry >= 0.01 * entry:
                        armed = True
                    if armed:
                        stop = max(stop, position * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                        exited = signed_price <= stop
                if exited:
                    position = 0
                    entry = np.nan
                    # Exit decisions remain flat for this close; no immediate reentry.
                    continue
            elif i >= warmup - 1 and candidates[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry = np.nan
                armed = False
            raw[i] = position
        df['signal'] = raw
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = EchoParams()
    k = GeneratedStrategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=k + 14, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 1000000.0}, index=index)
    computed = GeneratedStrategy.indicators(data, params)
    assert computed.index.equals(data.index)
    baseline = GeneratedStrategy.generate_signals(data, computed, None, params).data
    assert baseline['signal'].eq(0).all()
    for direction in (1, -1):
        sample = data.copy()
        sample.iloc[k + 2, sample.columns.get_loc('close')] = 100.0 + 2.0 * direction
        sample.iloc[k + 2, sample.columns.get_loc('high')] = max(100.5, 100.0 + 2.0 * direction)
        sample.iloc[k + 2, sample.columns.get_loc('low')] = min(99.5, 100.0 + 2.0 * direction)
        controlled = pd.DataFrame({'atr': 1.0, 'candidate': 0}, index=index)
        controlled.iloc[k, controlled.columns.get_loc('candidate')] = direction
        result = GeneratedStrategy.generate_signals(sample, controlled, None, params).data
        assert result['signal'].iloc[k] == 0
        assert result['signal'].iloc[k + 1] == direction
        assert result['signal'].iloc[k + 3] == direction
        assert result['signal'].iloc[k + 4] == 0
        assert result['signal'].isin([-1, 0, 1]).all()
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
