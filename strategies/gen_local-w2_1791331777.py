from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCompressionParams:
    energy_window: int = 12
    baseline_window: int = 40
    compression_ratio: float = 0.65
    fine_share: float = 0.60
    ma_band: float = 0.06


class GeneratedStrategy(BaseStrategy[WaveletCompressionParams]):
    strategy_id = 'gen_local-w2_1791331777'

    @classmethod
    def params_type(cls):
        return WaveletCompressionParams

    def warmup_bars(self, params: WaveletCompressionParams) -> int:
        return int(max(200, params.energy_window + params.baseline_window + 8))

    def indicators(self, data: pd.DataFrame, params: WaveletCompressionParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.baseline_window < 1:
            raise ValueError('Rolling windows must be positive.')
        if not 0 < params.compression_ratio < 1:
            raise ValueError('compression_ratio must lie between zero and one.')
        if not 0 < params.fine_share < 1 or params.ma_band <= 0:
            raise ValueError('Invalid energy concentration or MA corridor.')
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        for half in (1, 2, 4):
            scale = 2 * half
            out[f'mean_{half}'] = out['log_close'].rolling(half, min_periods=half).mean()
            out[f'haar_{scale}'] = (
                out[f'mean_{half}'] - out[f'mean_{half}'].shift(half)
            ) * np.sqrt(half / 2.0)
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out['total_energy'] = out['energy_2'] + out['energy_4'] + out['energy_8']
        out['baseline_energy'] = out['total_energy'].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out['fine_fraction'] = out['energy_2'] / out['total_energy'].where(out['total_energy'] > 0)
        out['ma_200'] = data['close'].rolling(200, min_periods=200).mean()
        out['ma_distance'] = data['close'] / out['ma_200'].where(out['ma_200'] > 0) - 1.0
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out['eligible'] = (
            (out['baseline_energy'] > 0)
            & (out['total_energy'] > 0)
            & (out['total_energy'] < params.compression_ratio * out['baseline_energy'])
            & (out['fine_fraction'] >= params.fine_share)
            & (out['ma_distance'].abs() <= params.ma_band)
        ).fillna(False)
        out['long_entry'] = (out['eligible'] & (out['haar_2'] < 0)).fillna(False)
        out['short_entry'] = (out['eligible'] & (out['haar_2'] > 0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletCompressionParams) -> SignalFrame:
        buys = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        sells = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (buy, sell) in enumerate(zip(buys, sells)):
            if (position == 1 and sell) or (position == -1 and buy):
                position = 0
            elif position == 0:
                position = 1 if buy else -1 if sell else 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'long_entry': [True, False, False, False, False, False],
        'short_entry': [False, False, True, True, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, WaveletCompressionParams()
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
