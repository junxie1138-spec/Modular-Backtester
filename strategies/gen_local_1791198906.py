from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    ma_window: int = 40
    pressure_window: int = 10
    slope_bars: int = 5
    z_threshold: float = 0.8
    pressure_threshold: float = 0.1
    holding_bars: int = 18

    def __post_init__(self):
        if self.ma_window < 2 or self.pressure_window < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if self.slope_bars < 1 or self.holding_bars < 1:
            raise ValueError('Slope and holding periods must be positive.')
        if not np.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError('z_threshold must be finite and positive.')
        if not np.isfinite(self.pressure_threshold) or not 0 < self.pressure_threshold < 1:
            raise ValueError('pressure_threshold must be between zero and one.')


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = 'gen_local_1791198906'

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.ma_window + params.slope_bars, params.pressure_window) + 1

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].replace(0.0, np.nan)
        out['ma_slope'] = out['ma'].diff(params.slope_bars)
        out['range'] = (data['high'] - data['low']).astype(float)
        out['intraday_pressure'] = ((close - data['open']) / out['range'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out.loc[out['range'].eq(0.0), 'intraday_pressure'] = 0.0
        out['pressure_volume'] = out['intraday_pressure'] * data['volume'].clip(lower=0)
        out['pressure_sum'] = out['pressure_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['volume_sum'] = data['volume'].clip(lower=0).rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['volume_pressure'] = out['pressure_sum'] / out['volume_sum'].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _fixed_positions(candidates: np.ndarray, holding_bars: int) -> np.ndarray:
        positions = np.zeros(len(candidates), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                positions[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + holding_bars
            positions[i] = position
        return positions

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorptionParams) -> SignalFrame:
        long_ready = ((indicators['z'] >= params.z_threshold)
                      & (indicators['ma_slope'] > 0)
                      & (indicators['volume_pressure'] <= -params.pressure_threshold))
        short_ready = ((indicators['z'] <= -params.z_threshold)
                       & (indicators['ma_slope'] < 0)
                       & (indicators['volume_pressure'] >= params.pressure_threshold))
        long_confirmed = long_ready & long_ready.shift(1, fill_value=False)
        short_confirmed = short_ready & short_ready.shift(1, fill_value=False)
        candidates = np.select([long_confirmed, short_confirmed], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = self._fixed_positions(candidates, params.holding_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    candidates = np.array([0, 1, -1, 0, -1, -1, 1, 0, 1, 0])
    raw = GeneratedStrategy._fixed_positions(candidates, 3)
    shifted = pd.Series(raw).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1, 0]
