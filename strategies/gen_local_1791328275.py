from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 24
    polarity_threshold: float = 0.66
    middle_share_threshold: float = 0.35
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791328275"

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return int(params.energy_window + 16)

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.hold_bars < 1:
            raise ValueError("Window must be at least two and holding period positive")
        if not 0.5 < params.polarity_threshold < 1.0:
            raise ValueError("Polarity threshold must lie between 0.5 and 1")
        if not 0.0 < params.middle_share_threshold < 1.0:
            raise ValueError("Middle share threshold must lie between 0 and 1")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        price = data['close'].astype(float)
        log_price = np.log(price.where(price > 0)).replace([np.inf, -np.inf], np.nan)
        out['log_price'] = log_price
        for name, half in (('fine', 2), ('middle', 4), ('coarse', 8)):
            average = log_price.rolling(half, min_periods=half).mean()
            coefficient = (average - average.shift(half)) * np.sqrt(half / 2.0)
            out[name + '_haar'] = coefficient
            out[name + '_energy'] = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        coarse = out['coarse_haar']
        out['coarse_up_energy'] = coarse.clip(lower=0).pow(2).rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean()
        out['total_energy'] = out[['fine_energy', 'middle_energy', 'coarse_energy']].sum(axis=1, min_count=3)
        out['polarity'] = out['coarse_up_energy'].div(out['coarse_energy'].where(out['coarse_energy'] > 0))
        out['middle_share'] = out['middle_energy'].div(out['total_energy'].where(out['total_energy'] > 0))
        out['fine_previous'] = out['fine_haar'].shift(1)
        ready = out[['polarity', 'middle_share', 'fine_haar', 'fine_previous', 'middle_haar', 'coarse_haar']].notna().all(axis=1)
        out['upturn'] = ready & out['fine_haar'].gt(0) & out['fine_previous'].le(0)
        out['trend_regime'] = ready & out['polarity'].ge(params.polarity_threshold)
        out['cycle_regime'] = ready & out['polarity'].between(
            1.0 - params.polarity_threshold, params.polarity_threshold, inclusive='neither'
        ) & out['middle_share'].ge(params.middle_share_threshold)
        trend_entry = out['trend_regime'] & out['coarse_haar'].gt(0)
        cycle_entry = out['cycle_regime'] & out['middle_haar'].lt(0)
        out['entry'] = out['upturn'] & (trend_entry | cycle_entry)
        out['strength'] = pd.Series(
            np.where(out['trend_regime'],
                     2.0 * out['polarity'] - 1.0,
                     out['middle_share'] * (1.0 - 2.0 * (out['polarity'] - 0.5).abs())),
            index=data.index
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        if params.hold_bars < 1 or not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Invalid holding period or position size")
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        entry_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                    continue
            elif entries[i]:
                entry_bar = i
                held_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
            if entry_bar >= 0:
                signal[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = WaveletParams()
    index = pd.date_range('2020-01-01', periods=20)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': True, 'strength': 0.4}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[:11].tolist() == [0] + [1] * 8 + [0, 1]
    expected_size = params.min_size + (params.max_size - params.min_size) * 0.4
    assert np.allclose(result.loc[result['signal'].eq(1), 'size'], expected_size)
    assert result['size'].gt(0).all()
