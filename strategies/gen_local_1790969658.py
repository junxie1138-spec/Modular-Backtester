from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    displacement_threshold: float = 1.3
    shock_ratio: float = 2.5
    profit_target: float = 0.035
    max_hold_bars: int = 18
    cooldown_bars: int = 5

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (
            self.displacement_threshold, self.shock_ratio, self.profit_target
        )):
            raise ValueError('Thresholds and profit_target must be finite and positive.')
        if self.max_hold_bars < 1 or self.cooldown_bars < 0:
            raise ValueError('Invalid holding period or cooldown.')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1790969658'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0))
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        for half in (2, 4, 8, 16):
            recent = out['log_close'].rolling(half, min_periods=half).mean()
            out[f'detail_{half}'] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f'energy_{half}'] = out[f'detail_{half}'].pow(2)
        out['fine_energy'] = out['energy_2'] + out['energy_4']
        out['middle_energy'] = out['energy_8'] + out['energy_16']
        out['total_energy'] = out['fine_energy'] + out['middle_energy']
        out['middle_share'] = out['middle_energy'] / out['total_energy'].where(out['total_energy'] > 0)
        out['fine_baseline'] = out['fine_energy'].rolling(60, min_periods=60).mean().shift(1)
        out['shock_strength'] = out['fine_energy'] / out['fine_baseline'].where(out['fine_baseline'] > 0)
        out['recent_shock'] = out['shock_strength'].shift(1).rolling(4, min_periods=4).max()
        out['displacement_scale'] = np.sqrt(out['energy_16'].rolling(60, min_periods=60).mean().shift(1))
        out['displacement'] = out['detail_16'] / out['displacement_scale'].where(out['displacement_scale'] > 0)
        out['previous_fine'] = out['fine_energy'].shift(1)
        out['previous_share'] = out['middle_share'].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out['energy_transfer'] = (
            ready
            & (out['recent_shock'] >= params.shock_ratio)
            & (out['fine_energy'] < 0.85 * out['previous_fine'])
            & (out['middle_share'] > out['previous_share'])
        )
        out['long_entry'] = (
            out['energy_transfer']
            & (out['displacement'] <= -params.displacement_threshold)
            & (close > out['ma200'])
        )
        out['short_entry'] = (
            out['energy_transfer']
            & (out['displacement'] >= params.displacement_threshold)
            & (close < out['ma200'])
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        next_entry_bar = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if side:
                # Read the entry open only when its execution bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or gain >= params.profit_target or expired:
                    side = 0
                    entry_price = np.nan
                    next_entry_bar = i + params.cooldown_bars + 1
                signal[i] = side
                continue
            if i >= next_entry_bar and np.isfinite(closes[i]) and closes[i] > 0:
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
                if side:
                    entry_bar = i + 1
                    entry_price = np.nan
            signal[i] = side
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=240, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 105.0, 'low': 95.0,
        'close': 100.0, 'volume': 1000000.0,
    }, index=index)
    data.iloc[201, data.columns.get_loc('close')] = 104.0
    data.iloc[211, data.columns.get_loc('close')] = 96.0
    entries = pd.DataFrame({'long_entry': False, 'short_entry': False}, index=index)
    entries.iloc[200, 0] = True
    entries.iloc[210, 1] = True
    entries.iloc[220, 0] = True
    strategy = GeneratedStrategy()
    params = WaveletParams()
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].iloc[200] == 0
    assert result['signal'].iloc[201] == 1 and result['signal'].iloc[202] == 0
    assert result['signal'].iloc[211] == -1 and result['signal'].iloc[212] == 0
    assert (result['signal'].iloc[221:239] == 1).all()
    assert result['signal'].iloc[239] == 0
    assert result['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived['long_entry'].iloc[:199].any()
    assert not derived['short_entry'].iloc[:199].any()
