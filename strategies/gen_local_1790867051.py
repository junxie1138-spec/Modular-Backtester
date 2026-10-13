from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletMigrationParams:
    percentile_window: int = 84
    transfer_quantile: float = 0.8
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.8
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[WaveletMigrationParams]):
    strategy_id = 'gen_local_1790867051'

    @classmethod
    def params_type(cls):
        return WaveletMigrationParams

    def warmup_bars(self, params: WaveletMigrationParams) -> int:
        return int(max(params.percentile_window + 21, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletMigrationParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.atr_window < 1:
            raise ValueError('Lookback windows must be positive; percentile_window must exceed one.')
        if not 0.0 < params.transfer_quantile < 1.0:
            raise ValueError('transfer_quantile must be between zero and one.')
        if params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0:
            raise ValueError('Stop parameters must be positive.')
        if params.max_hold not in (3, 4, 5):
            raise ValueError('max_hold must be three, four, or five sessions.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['log_return'] = np.log(close).diff()
        for scale in (2, 4, 8):
            mean = out['log_return'].rolling(scale, min_periods=scale).mean()
            out[f'mean_{scale}'] = mean
            out[f'haar_{scale}'] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(3, min_periods=3).mean()
        out['total_energy'] = out['energy_2'] + out['energy_4'] + out['energy_8']
        out['coarse_share'] = out['energy_8'] / out['total_energy'].where(out['total_energy'] > 0.0)
        out['transfer'] = out['coarse_share'].diff(2)
        window = params.percentile_window
        out['transfer_threshold'] = out['transfer'].shift(1).rolling(window, min_periods=window).quantile(params.transfer_quantile)
        out['activity_threshold'] = out['total_energy'].shift(1).rolling(window, min_periods=window).quantile(0.5)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = out[['transfer', 'transfer_threshold', 'total_energy', 'activity_threshold', 'haar_4', 'haar_8', 'mean_8', 'atr']].notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['transfer'] > out['transfer_threshold'])
            & (out['transfer'] > 0.0)
            & (out['total_energy'] > out['activity_threshold'])
            & (out['haar_4'] > 0.0)
            & (out['haar_8'] > 0.0)
            & (out['mean_8'] > 0.0)
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletMigrationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry_price = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                # The preceding close's decision fills at this session's open.
                entry_price = opens[i]
                prior_atr = atr[i - 1] if i else np.nan
                active = bool(np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(prior_atr) and prior_atr > 0.0)
                stop = entry_price - params.trail_atr * prior_atr if active else 0.0
                armed = False
                age = 0
                pending = False
                if not active:
                    continue
            if active:
                age += 1
                valid = np.isfinite(closes[i]) and np.isfinite(lows[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
                # A breach is observed at the close; liquidation fills next open.
                # Test the previously established stop before raising it.
                if not valid or lows[i] <= stop or age >= params.max_hold:
                    active = False
                    continue
                if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, closes[i] - params.trail_atr * atr[i])
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                decisions[i] = 1
                pending = True
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [100.2, 101.5, 103.4, 102.0, 100.2, 100.2],
        'low': [99.9, 99.8, 100.8, 100.9, 99.9, 99.9],
        'close': [100.0, 101.2, 103.0, 101.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 6, 'entry': [True, False, False, False, False, False]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    params = WaveletMigrationParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    data['close'] = 100.0
    data['low'] = 99.9
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    indicators['entry'] = False
    indicators['atr'] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
