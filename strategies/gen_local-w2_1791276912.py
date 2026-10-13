from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    slope_bars: int = 5
    noise_bars: int = 30
    entry_z: float = 1.8
    center_limit: float = 0.6
    profit_target: float = 0.008
    hold_bars: int = 4
    min_size: float = 0.25

    def __post_init__(self):
        if self.slope_bars < 2 or self.noise_bars < 3:
            raise ValueError('Lookbacks must be at least 2 and 3 bars.')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be 3, 4, or 5.')
        values = (self.entry_z, self.center_limit, self.profit_target, self.min_size)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds and sizing must be positive and finite.')
        if self.min_size > 1:
            raise ValueError('min_size must not exceed 1.')


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = 'gen_local-w2_1791276912'

    @classmethod
    def params_type(cls):
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return int(params.slope_bars + params.noise_bars + 2)

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        midpoint = (data['high'] + data['low']) / 2.0
        out['log_midpoint'] = np.log(midpoint.where(midpoint > 0))
        out['close_return'] = out['log_close'].diff()
        out['midpoint_return'] = out['log_midpoint'].diff()
        out['latent_slope'] = out['midpoint_return'].rolling(
            params.slope_bars, min_periods=params.slope_bars
        ).mean()
        out['predicted_slope'] = out['latent_slope'].shift(1)
        out['innovation'] = out['close_return'] - out['predicted_slope']
        out['center_innovation'] = out['midpoint_return'] - out['predicted_slope']
        out['noise'] = out['innovation'].rolling(
            params.noise_bars, min_periods=params.noise_bars
        ).std(ddof=0).shift(1)
        denominator = out['noise'].where(out['noise'] > 1e-12)
        out['z'] = out['innovation'] / denominator
        out['center_z'] = out['center_innovation'] / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        z = indicators['z'].to_numpy(dtype=float)
        center_z = indicators['center_z'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        eligible = (
            np.isfinite(z)
            & np.isfinite(center_z)
            & (np.abs(z) >= params.entry_z)
            & (np.abs(center_z) <= params.center_limit)
            & np.isfinite(closes)
            & (closes > 0)
        )
        strength = np.clip((np.abs(z) - params.entry_z) / params.entry_z, 0, 1)
        candidate_size = params.min_size + (1 - params.min_size) * strength
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        position_size = params.min_size

        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid_entry or gain >= params.profit_target or expired:
                    # Close-confirmed exits fill at the following open.
                    position = 0
                    entry_price = np.nan
                else:
                    signals[i] = position
                    sizes[i] = position_size
                # Do not reopen on the same decision bar as an exit.
                continue

            if eligible[i]:
                position = -1 if z[i] > 0 else 1
                entry_bar = i + 1
                entry_price = np.nan
                position_size = float(candidate_size[i])
                signals[i] = position
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = InnovationParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': 100.0, 'high': 102.0, 'low': 98.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    ind = pd.DataFrame({'z': np.nan, 'center_z': np.nan}, index=index)
    ind.iloc[0] = [-3.6, 0.0]
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result['size'].iloc[1] == 1.0
    assert (result['size'] > 0).all()
    for direction in (1, -1):
        sample = data.copy()
        sample.loc[index[1], 'close'] = 100.0 * (1 + direction * 0.01)
        ind.iloc[0] = [-direction * 2.7, 0.0]
        result = strategy.generate_signals(sample, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, 0, 0, 0, 0, 0, 0]
        assert params.min_size < result['size'].iloc[1] < 1.0
    flat = strategy.generate_signals(data, ind * np.nan, None, params).data
    assert flat['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
