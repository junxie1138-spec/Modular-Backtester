from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletMigrationParams:
    baseline_window: int = 60
    energy_window: int = 4
    change_threshold: float = 0.10
    energy_multiple: float = 1.20
    holding_bars: int = 18

    def __post_init__(self):
        if self.baseline_window < 2 or self.energy_window < 2:
            raise ValueError('Energy windows must be at least two bars.')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive.')
        if not 0.0 < self.change_threshold <= 2.0:
            raise ValueError('change_threshold must be in (0, 2].')
        if not np.isfinite(self.energy_multiple) or self.energy_multiple <= 0:
            raise ValueError('energy_multiple must be finite and positive.')


class GeneratedStrategy(BaseStrategy[WaveletMigrationParams]):
    strategy_id = 'gen_local_1790850782'

    @classmethod
    def params_type(cls) -> type[WaveletMigrationParams]:
        return WaveletMigrationParams

    def warmup_bars(self, params: WaveletMigrationParams) -> int:
        # Eight return observations, their preceding close, and both windows.
        return int(8 + params.baseline_window + params.energy_window)

    def indicators(self, data: pd.DataFrame, params: WaveletMigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_close'] = np.log(close).replace([np.inf, -np.inf], np.nan)
        out['return'] = out['log_close'].diff()
        recent_columns = []
        baseline_columns = []
        for scale in (1, 2, 4):
            mean = out['return'].rolling(scale, min_periods=scale).mean()
            out[f'mean_{scale}'] = mean
            out[f'haar_{scale}'] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2)
            recent = f'recent_{scale}'
            baseline = f'baseline_{scale}'
            out[recent] = out[f'energy_{scale}'].rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            # The baseline ends before the entire recent-energy window.
            out[baseline] = out[f'energy_{scale}'].rolling(
                params.baseline_window, min_periods=params.baseline_window
            ).mean().shift(params.energy_window)
            recent_columns.append(recent)
            baseline_columns.append(baseline)
        out['recent_total'] = out[recent_columns].sum(axis=1, min_count=3)
        out['baseline_total'] = out[baseline_columns].sum(axis=1, min_count=3)
        for scale in (1, 2, 4):
            out[f'recent_share_{scale}'] = out[f'recent_{scale}'] / out['recent_total'].replace(0.0, np.nan)
            out[f'baseline_share_{scale}'] = out[f'baseline_{scale}'] / out['baseline_total'].replace(0.0, np.nan)
        changes = pd.DataFrame({
            scale: (out[f'recent_share_{scale}'] - out[f'baseline_share_{scale}']).pow(2)
            for scale in (1, 2, 4)
        }, index=data.index)
        out['spectral_change'] = changes.sum(axis=1, min_count=3)
        out['fine_share_drop'] = out['baseline_share_1'] - out['recent_share_1']
        out['energy_expansion'] = out['recent_total'] / out['baseline_total'].replace(0.0, np.nan)
        out['momentum_4'] = out['return'].rolling(4, min_periods=4).sum()
        out['momentum_8'] = out['return'].rolling(8, min_periods=8).sum()
        out['setup'] = (
            (out['spectral_change'] >= params.change_threshold)
            & (out['fine_share_drop'] >= 0.10)
            & (out['energy_expansion'] >= params.energy_multiple)
            & (out['momentum_4'] > 0.0)
            & (out['momentum_8'] > 0.0)
        ).fillna(False)
        out['confirmed'] = out['setup'] & out['setup'].shift(1, fill_value=False)
        out['entry_event'] = out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletMigrationParams,
    ) -> SignalFrame:
        entries = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i >= exit_bar:
                    exit_bar = -1
                    # Always emit a flat bar at expiry, even with a new setup.
                    continue
                signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # A decision on bar 2 fills on bar 3 and exits exactly 18 bars later.
    params = WaveletMigrationParams()
    index = pd.date_range('2020-01-01', periods=30)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry_event': False}, index=index)
    indicators.iloc[2, 0] = True
    indicators.iloc[10, 0] = True
    indicators.iloc[20, 0] = True
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    expected = np.zeros(30, dtype=int)
    expected[3:21] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert (result['size'] > 0.0).all()
