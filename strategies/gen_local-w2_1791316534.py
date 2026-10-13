from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveTransferParams:
    energy_window: int = 20
    transfer_min: float = 0.15
    pressure_window: int = 3
    pressure_min: float = 0.15
    atr_window: int = 14
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveTransferParams]):
    strategy_id = 'gen_local-w2_1791316534'

    @classmethod
    def params_type(cls) -> type[WaveTransferParams]:
        return WaveTransferParams

    @staticmethod
    def warmup_bars(params: WaveTransferParams) -> int:
        return int(max(16 + params.energy_window + 1,
                       params.atr_window + 1,
                       2 * params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveTransferParams) -> pd.DataFrame:
        if (params.energy_window < 2 or params.pressure_window < 1
                or params.atr_window < 1 or params.stop_atr <= 0
                or params.hold_bars not in (1, 2)
                or not 0 < params.transfer_min < 1
                or not 0 <= params.pressure_min <= 1):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data['close'].where(data['close'] > 0))
        out['log_close'] = log_close
        for scale in (2, 4, 8):
            block = log_close.rolling(scale, min_periods=scale).mean()
            coefficient = (block - block.shift(scale)) * np.sqrt(scale / 2.0)
            out[f'wave_{scale}'] = coefficient
            out[f'energy_{scale}'] = coefficient.pow(2)
        total = out[['energy_2', 'energy_4', 'energy_8']].sum(axis=1, min_count=3)
        denominator = total.where(total > 0)
        out['total_energy'] = total
        out['fine_share'] = out['energy_2'] / denominator
        out['coarse_share'] = out['energy_8'] / denominator
        out['energy_baseline'] = total.rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean().shift(1)
        out['fine_transfer'] = out['fine_share'] - out['fine_share'].shift(1)
        out['coarse_transfer'] = out['coarse_share'] - out['coarse_share'].shift(1)
        out['wave_agrees'] = (
            (out['fine_transfer'] >= params.transfer_min)
            & (out['coarse_transfer'] <= -params.transfer_min)
            & (out['coarse_share'].shift(1) >= 0.45)
            & (out['wave_8'].shift(1) > 0)
            & (out['wave_2'] > 0)
            & (total > 0.5 * out['energy_baseline'])
        ).fillna(False)
        spread = data['high'] - data['low']
        out['close_location'] = (
            (2 * data['close'] - data['high'] - data['low'])
            / spread.where(spread > 0)
        ).clip(-1, 1).where(spread != 0, 0.0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['valid_volume'] = volume
        out['pressure_flow'] = out['close_location'] * volume
        out['pressure_volume'] = volume.rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out['pressure'] = out['pressure_flow'].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum() / out['pressure_volume'].where(out['pressure_volume'] > 0)
        out['previous_pressure'] = out['pressure'].shift(params.pressure_window)
        out['pressure_agrees'] = (
            (out['pressure'] > params.pressure_min)
            & (out['pressure'] > out['previous_pressure'])
        ).fillna(False)
        previous_close = data['close'].shift(1)
        out['previous_close'] = previous_close
        out['true_range'] = pd.concat([
            spread,
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            out['wave_agrees'].fillna(False)
            & out['pressure_agrees'].fillna(False)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveTransferParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        age = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if (np.isfinite(opening[i]) and opening[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    # Entry fills at this open; freeze volatility from the decision bar.
                    fixed_stop = opening[i] - params.stop_atr * atr[i - 1]
                    active = True
                    age = 0
            if active:
                age += 1
                if (not np.isfinite(close[i]) or close[i] < fixed_stop
                        or age >= params.hold_bars):
                    active = False
                else:
                    raw[i] = 1
            elif entry[i] and np.isfinite(close[i]) and close[i] > 0:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    sample = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [101.0, 101.0, 101.0, 111.0, 101.0, 101.0],
        'low': [99.0, 96.0, 99.0, 99.0, 98.0, 99.0],
        'close': [100.0, 97.0, 100.0, 110.0, 99.0, 100.0],
        'volume': [1000.0] * 6
    }, index=index)
    supplied = pd.DataFrame({
        'atr': [2.0] * 6,
        'entry': [True, False, True, False, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, supplied, None, WaveTransferParams())
    assert result.data['signal'].tolist() == [0, 1, 0, 1, 1, 0]
    assert result.data['size'].gt(0).all()
