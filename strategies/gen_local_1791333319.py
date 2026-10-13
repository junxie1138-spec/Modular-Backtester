from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDrawdownParams:
    peak_window: int = 20
    atr_window: int = 14
    min_drawdown: float = 0.03
    min_gap: float = 0.001
    repair_fraction: float = 0.5
    breakeven_gain: float = 0.004
    trail_atr: float = 0.8
    initial_stop_atr: float = 2.0

    def __post_init__(self):
        if self.peak_window < 2 or self.atr_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        values = (self.min_drawdown, self.min_gap, self.repair_fraction,
                  self.breakeven_gain, self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('Thresholds and ATR multiples must be finite and positive.')
        if self.min_drawdown >= 1 or self.min_gap >= 1:
            raise ValueError('Drawdown and gap thresholds must be below one.')


class GeneratedStrategy(BaseStrategy[GapDrawdownParams]):
    strategy_id = 'gen_local_1791333319'

    @classmethod
    def params_type(cls):
        return GapDrawdownParams

    def warmup_bars(self, params: GapDrawdownParams) -> int:
        return int(max(params.peak_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapDrawdownParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['peak'] = data['close'].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max().shift(1)
        previous = out['previous_close'].where(out['previous_close'] > 0)
        peak = out['peak'].where(out['peak'] > 0)
        out['gap'] = data['open'] / previous - 1.0
        out['opening_drawdown'] = 1.0 - data['open'] / peak
        out['closing_drawdown'] = 1.0 - data['close'] / peak
        gap_loss = (previous - data['open']).where(previous > data['open'])
        out['repair_fraction'] = (data['close'] - data['open']) / gap_loss
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = (
            out[['peak', 'gap', 'opening_drawdown', 'closing_drawdown',
                 'repair_fraction', 'atr']].notna().all(axis=1)
            & out['atr'].gt(0)
            & data[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
            & np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        )
        out['qualified'] = (
            out['ready']
            & out['gap'].le(-params.min_gap)
            & out['opening_drawdown'].ge(params.min_drawdown)
            & out['closing_drawdown'].gt(0)
            & out['repair_fraction'].ge(params.repair_fraction)
        )
        out['confirmed'] = (
            out['qualified']
            & out['qualified'].shift(1, fill_value=False)
            & out['opening_drawdown'].lt(out['opening_drawdown'].shift(1))
            & out['closing_drawdown'].lt(out['closing_drawdown'].shift(1))
            & out['peak'].ge(out['peak'].shift(1))
        )
        out['entry_event'] = (
            out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapDrawdownParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        pending = False
        held = False
        protected = False
        entry = stop = watermark = 0.0
        age = 0

        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and (prices[i] > 0).all()
            if not valid:
                pending = held = False
                continue

            if pending:
                pending = False
                if i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                # Entry price becomes available only on the next bar's open.
                entry = watermark = opening
                stop = max(0.0, entry - params.initial_stop_atr * atr[i - 1])
                protected = False
                age = 0
                held = True

            if held:
                age += 1
                # Test the existing stop before using this bar's close to ratchet it.
                # A breach requests an exit at the following open, not at the stop price.
                if low <= stop or age >= 2:
                    held = False
                    continue
                watermark = max(watermark, close)
                if close >= entry * (1.0 + params.breakeven_gain):
                    protected = True
                if protected:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, watermark - params.trail_atr * atr[i])
                if close <= stop:
                    held = False
                    continue
                raw[i] = 1
            elif entries[i]:
                pending = True
                raw[i] = 1

        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def demo():
    params = GapDrawdownParams(peak_window=3, atr_window=2, min_drawdown=0.02)
    opening = np.array([100, 100, 100, 96, 94, 95, 96.8, 97.2, 97.0])
    close = np.array([100, 100, 100, 95, 96, 96.8, 97.6, 97.0, 97.1])
    data = pd.DataFrame({
        'open': opening, 'high': np.maximum(opening, close) + 0.2,
        'low': np.minimum(opening, close) - 0.2,
        'close': close, 'volume': 1000000.0
    }, index=pd.date_range('2020-01-01', periods=len(close)))
    data.iloc[7, data.columns.get_loc('low')] = 96.5
    strategy = object.__new__(GeneratedStrategy)
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert derived['entry_event'].tolist() == [False] * 5 + [True, False, False, False]
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
    for end in range(1, len(data) + 1):
        prefix = data.iloc[:end]
        partial = strategy.generate_signals(
            prefix, strategy.indicators(prefix, params), None, params
        ).data
        assert partial['signal'].equals(result['signal'].iloc[:end])


if __name__ == '__main__':
    demo()
