from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 5
    compression_window: int = 80
    compression_quantile: float = 0.25
    setup_window: int = 10
    coarse_share: float = 0.65
    atr_period: int = 14
    activation_pct: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791271474'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, 16 + params.energy_window - 1 + params.compression_window - 1 + params.setup_window + 1, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.energy_window, params.compression_window, params.setup_window, params.atr_period, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0 < params.compression_quantile < 1 or not 0 < params.coarse_share < 1:
            raise ValueError('Quantile and energy share must lie between zero and one.')
        if params.activation_pct <= 0 or params.trail_atr <= 0:
            raise ValueError('Activation and ATR multiplier must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0))
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        # Causal, undecimated Haar details: adjacent equal-length blocks.
        for scale in (4, 8, 16):
            half = scale // 2
            out[f'mean_{half}'] = out['log_close'].rolling(half, min_periods=half).mean()
            out[f'haar_{scale}'] = np.sqrt(half / 2.0) * (out[f'mean_{half}'] - out[f'mean_{half}'].shift(half))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out['total_energy'] = out[['energy_4', 'energy_8', 'energy_16']].sum(axis=1, min_count=3)
        out['coarse_fraction'] = out['energy_16'].div(out['total_energy'].where(out['total_energy'] > 0))
        out['range_high'] = data['high'].rolling(8, min_periods=8).max()
        out['range_low'] = data['low'].rolling(8, min_periods=8).min()
        out['range_fraction'] = (out['range_high'] - out['range_low']).div(close.where(close > 0))
        out['energy_floor'] = out['total_energy'].shift(1).rolling(params.compression_window, min_periods=params.compression_window).quantile(params.compression_quantile)
        out['range_floor'] = out['range_fraction'].shift(1).rolling(params.compression_window, min_periods=params.compression_window).quantile(params.compression_quantile)
        out['compressed'] = ((out['total_energy'] <= out['energy_floor']) & (out['range_fraction'] <= out['range_floor'])).fillna(False)
        out['armed'] = out['compressed'].astype(float).shift(1).rolling(params.setup_window, min_periods=params.setup_window).max().fillna(0).eq(1)
        out['breakout_level'] = out['range_high'].shift(1)
        out['entry'] = (
            out['armed']
            & (close > out['ma200'])
            & (close > out['breakout_level'])
            & (out['haar_16'] > 0)
            & (out['coarse_fraction'] >= params.coarse_share)
            & (out['coarse_fraction'] > out['coarse_fraction'].shift(1))
            & (out['atr'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        active = False
        protected = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            # The preceding close's decision enters at this bar's open.
            if not active and i > 0 and desired[i - 1] == 1:
                entry = float(data['open'].iloc[i])
                previous_atr = float(indicators['atr'].iloc[i - 1])
                if np.isfinite(entry) and entry > 0 and np.isfinite(previous_atr) and previous_atr > 0:
                    active = True
                    protected = False
                    age = 0
                    peak = entry
                    stop = entry - params.trail_atr * previous_atr
                else:
                    continue
            close = float(data['close'].iloc[i])
            low = float(data['low'].iloc[i])
            atr = float(indicators['atr'].iloc[i])
            if active:
                age += 1
                # Stop breaches become exit decisions filled next session.
                if not np.isfinite(close) or not np.isfinite(low) or not np.isfinite(atr) or atr <= 0:
                    active = False
                    continue
                if low <= stop or age >= params.hold_bars:
                    active = False
                    continue
                peak = max(peak, close)
                if close >= entry * (1.0 + params.activation_pct):
                    protected = True
                if protected:
                    stop = max(stop, entry, peak - params.trail_atr * atr)
                    if close <= stop:
                        active = False
                        continue
                desired[i] = 1
            elif bool(indicators['entry'].iloc[i]):
                desired[i] = 1
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100, 100, 104, 108, 107, 106],
        'high': [101, 105, 109, 109, 108, 107],
        'low': [99, 99, 103, 106, 106, 105],
        'close': [100, 104, 108, 107, 107, 106],
        'volume': [1000] * 6,
    }, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'entry': [True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(Params()) >= 200


if __name__ == '__main__':
    _self_check()
