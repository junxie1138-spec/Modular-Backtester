from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletOverflowParams:
    breakout_bars: int = 3
    release_share: float = 0.45
    stored_share: float = 0.20


class GeneratedStrategy(BaseStrategy[WaveletOverflowParams]):
    strategy_id = 'gen_local-w2_1791245576'

    @classmethod
    def params_type(cls) -> type[WaveletOverflowParams]:
        return WaveletOverflowParams

    def warmup_bars(self, params: WaveletOverflowParams) -> int:
        if not 2 <= params.breakout_bars <= 8:
            raise ValueError('breakout_bars must be between 2 and 8')
        if not 0 < params.stored_share < params.release_share < 1:
            raise ValueError('Require 0 < stored_share < release_share < 1')
        return max(9, params.breakout_bars + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletOverflowParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        x = out['log_close']
        out['haar_2'] = (x - x.shift(1)) / np.sqrt(2.0)
        out['haar_4'] = (x + x.shift(1) - x.shift(2) - x.shift(3)) / 2.0
        out['haar_8'] = (
            x + x.shift(1) + x.shift(2) + x.shift(3)
            - x.shift(4) - x.shift(5) - x.shift(6) - x.shift(7)
        ) / np.sqrt(8.0)
        out['energy_2'] = out['haar_2'].pow(2)
        out['energy_4'] = out['haar_4'].pow(2)
        out['energy_8'] = out['haar_8'].pow(2)
        out['total_energy'] = out['energy_2'] + out['energy_4'] + out['energy_8']
        out['fine_share'] = out['energy_2'] / out['total_energy'].replace(0.0, np.nan)
        out['previous_fine_share'] = out['fine_share'].shift(1)
        out['upper_channel'] = data['high'].shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max()
        out['lower_channel'] = data['low'].shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).min()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['release_setup'] = (
            out['previous_fine_share'].le(params.stored_share)
            & out['fine_share'].ge(params.release_share)
        ).fillna(False)
        out['long_condition'] = (
            out['fine_share'].ge(params.release_share)
            & data['close'].gt(out['upper_channel'])
            & np.isfinite(data['close'])
        ).fillna(False)
        out['short_condition'] = (
            out['fine_share'].ge(params.release_share)
            & data['close'].lt(out['lower_channel'])
            & np.isfinite(data['close'])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletOverflowParams,
    ) -> SignalFrame:
        direction = np.select(
            [indicators['long_condition'].fillna(False).to_numpy(dtype=bool),
             indicators['short_condition'].fillna(False).to_numpy(dtype=bool)],
            [1, -1], default=0,
        )
        setups = indicators['release_setup'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            # Exit only when the held direction's entry condition flips false.
            # The exit bar is flat even if an opposing entry qualifies.
            if position != 0 and direction[i] != position:
                position = 0
            elif position == 0 and setups[i]:
                position = int(direction[i])
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = WaveletOverflowParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'release_setup': [True, False, False, True, False],
        'long_condition': [True, True, False, False, False],
        'short_condition': [False, False, False, True, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) == 9
    prices = np.array([100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 104, 104, 104, 104])
    bars = pd.DataFrame({
        'open': prices, 'high': prices + 0.05, 'low': prices - 0.05,
        'close': prices, 'volume': 1000000.0,
    }, index=pd.date_range('2020-01-01', periods=len(prices)))
    calculated = strategy.indicators(bars, params)
    assert not calculated['release_setup'].iloc[:8].any()
    assert calculated['release_setup'].iloc[8]
    assert calculated['long_condition'].iloc[8]


if __name__ == '__main__':
    _self_check()
