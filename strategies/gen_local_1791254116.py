from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    flow_window: int = 3
    baseline_window: int = 20
    participation_jump: float = 0.15
    floor_bars: int = 3
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    max_hold_bars: int = 5

    def __post_init__(self):
        if min(self.flow_window, self.baseline_window, self.atr_window, self.max_hold_bars) < 1 or self.floor_bars < 2:
            raise ValueError('Lookbacks and holding period must be positive; floor_bars must be at least two.')
        if not 0.0 <= self.participation_jump <= 1.0:
            raise ValueError('participation_jump must be between zero and one.')
        if not all(np.isfinite(x) and x > 0 for x in (self.initial_stop_atr, self.breakeven_pct, self.trail_atr)):
            raise ValueError('Stop parameters must be finite and positive.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791254116'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.flow_window + params.baseline_window + 1, params.atr_window + 1, params.floor_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['close_change'] = data['close'].diff()
        out['low_change'] = data['low'].diff()
        out['volume'] = data['volume'].where(data['volume'] >= 0)
        out['advancing_volume'] = out['volume'].where(out['close_change'] > 0, 0.0).where(out['close_change'].notna())
        out['recent_volume'] = out['volume'].rolling(params.flow_window).sum()
        out['recent_advancing_volume'] = out['advancing_volume'].rolling(params.flow_window).sum()
        out['baseline_volume'] = out['volume'].rolling(params.baseline_window).sum().shift(params.flow_window)
        out['baseline_advancing_volume'] = out['advancing_volume'].rolling(params.baseline_window).sum().shift(params.flow_window)
        out['recent_share'] = out['recent_advancing_volume'] / out['recent_volume'].replace(0.0, np.nan)
        out['baseline_share'] = out['baseline_advancing_volume'] / out['baseline_volume'].replace(0.0, np.nan)
        out['participation_change'] = out['recent_share'] - out['baseline_share']
        out['rising_low_count'] = out['low_change'].gt(0).astype(float).where(out['low_change'].notna()).rolling(params.floor_bars - 1).sum()
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['entry'] = (
            out['participation_change'].ge(params.participation_jump)
            & out['recent_share'].gt(0.5)
            & out['rising_low_count'].eq(params.floor_bars - 1)
            & out['close_change'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        peak_close = 0.0
        stop = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    active = True
                    armed = False
                    entry_price = opens[i]
                    peak_close = entry_price
                    entry_bar = i
                    stop = entry_price - params.initial_stop_atr * atrs[i - 1]

            if active:
                # A touched stop requests liquidation at the following open.
                # Check the previously established stop before updating it.
                expired = i - entry_bar + 1 >= params.max_hold_bars
                touched = np.isfinite(lows[i]) and lows[i] <= stop
                invalid = not (np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0)
                if touched or expired or invalid:
                    active = False
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    # Close-based activation avoids assuming intrabar high/low order.
                    stop = max(stop, entry_price, peak_close - params.trail_atr * atrs[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                targets[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                targets[i] = 1
                pending = True

        df = pd.DataFrame({'signal': targets, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [100.0] * 8,
        'high': [101.0, 103.0, 105.0, 105.0, 101.0, 101.0, 101.0, 101.0],
        'low': [99.0, 100.0, 101.0, 102.0, 99.0, 99.0, 99.0, 99.0],
        'close': [100.0, 102.0, 104.0, 103.0, 100.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 8,
    }, index=index)
    derived = pd.DataFrame({'atr': [1.0] * 8, 'entry': [True] + [False] * 7}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    # Activation raises the stop to 100.5; the next close raises it to 102.5.
    # The following low touches that ratchet, producing next-bar liquidation.
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = [100.0, 100.5, 99.5, 100.0]
    result = strategy.generate_signals(flat, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    blank = strategy.indicators(flat, params)
    assert not blank['entry'].any()
    assert strategy.warmup_bars(params) >= params.flow_window + params.baseline_window + 1


if __name__ == '__main__':
    _self_check()
