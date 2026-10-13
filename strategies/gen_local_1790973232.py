from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    recent_window: int = 6
    reference_window: int = 24
    change_threshold: float = 0.35
    entropy_drop: float = 0.15
    order_gain: float = 0.15
    atr_period: int = 14
    breakeven_pct: float = 0.008
    initial_atr: float = 2.0
    trail_atr: float = 1.6
    max_hold: int = 4
    min_size: float = 0.25
    max_size: float = 0.95
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1790973232'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.reference_window + params.recent_window + 2,
                       params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (params.recent_window >= 3 and
                params.reference_window >= params.recent_window and
                params.atr_period >= 1 and 3 <= params.max_hold <= 5 and
                0 < params.change_threshold < 1 and
                0 < params.entropy_drop < 1 and
                0 < params.order_gain < 1 and
                params.breakeven_pct > 0 and
                params.initial_atr > 0 and params.trail_atr > 0 and
                0 < params.min_size <= params.max_size <= 1 and
                params.entry_slippage_bps >= 0):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Older observations win ties, giving a deterministic total ordering.
        out['pattern'] = ((a <= b).astype(int) * 4 +
                          (b <= c).astype(int) * 2 +
                          (a <= c).astype(int)).where(valid)
        recent_columns, reference_columns = [], []
        for code in (0, 1, 3, 4, 6, 7):
            name = f'pattern_{code}'
            recent_name, reference_name = f'recent_{code}', f'reference_{code}'
            out[name] = (out['pattern'] == code).astype(float).where(valid)
            out[recent_name] = out[name].rolling(params.recent_window).mean()
            out[reference_name] = out[name].shift(params.recent_window).rolling(
                params.reference_window).mean()
            recent_columns.append(recent_name)
            reference_columns.append(reference_name)
        recent = out[recent_columns].to_numpy(dtype=float)
        reference = out[reference_columns].to_numpy(dtype=float)
        out['recent_entropy'] = -np.sum(
            recent * np.log(np.clip(recent, 1e-12, 1.0)), axis=1) / np.log(6.0)
        out['reference_entropy'] = -np.sum(
            reference * np.log(np.clip(reference, 1e-12, 1.0)), axis=1) / np.log(6.0)
        out['distribution_change'] = 0.5 * np.sum(np.abs(recent - reference), axis=1)
        out['entropy_drop'] = out['reference_entropy'] - out['recent_entropy']
        out['order_gain'] = out['recent_7'] - out['reference_7']
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - close.shift(1)).abs(),
            (data['low'] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period).mean()
        out['conviction'] = (
            0.5 * out['distribution_change'] +
            0.3 * out['entropy_drop'].clip(0, 1) +
            0.2 * out['order_gain'].clip(0, 1)
        ).clip(0, 1).fillna(0.0)
        out['eligible'] = (
            (out['distribution_change'] >= params.change_threshold) &
            (out['entropy_drop'] >= params.entropy_drop) &
            (out['order_gain'] >= params.order_gain) &
            (close > close.shift(1)) &
            np.isfinite(out['atr']) & (out['atr'] > 0)
        ).fillna(False)
        out['entry'] = out['eligible'] & ~out['eligible'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.min_size, dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].fillna(0).to_numpy(dtype=float)
        active, armed = False, False
        entry_price, entry_atr = np.nan, np.nan
        stop, peak, allocation = 0.0, 0.0, params.min_size
        entered_at = -1
        for i in range(n):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and (prices[i] > 0).all()
            valid = valid and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                active = False
                entry_price = np.nan
                continue
            if active:
                # The prior close scheduled entry; today's open is now observable.
                if np.isnan(entry_price):
                    entry_price = opening * (1.0 + params.entry_slippage_bps / 10000.0)
                    entered_at = i
                    peak = entry_price
                    stop = max(0.0, entry_price - params.initial_atr * entry_atr)
                    armed = False
                breached = low <= stop
                if not breached:
                    peak = max(peak, high)
                    if high >= entry_price * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                    # Test new ratchets against the close; daily OHLC cannot
                    # establish whether today's low preceded today's high.
                    breached = close <= stop
                if breached or i - entered_at + 1 >= params.max_hold:
                    active = False
                    entry_price = np.nan
                else:
                    raw_signal[i] = 1
                raw_size[i] = allocation
                continue
            if entries[i]:
                active = True
                entry_price = np.nan
                entry_atr = atr[i]
                allocation = params.min_size + (params.max_size - params.min_size) * float(
                    np.clip(conviction[i], 0.0, 1.0))
                raw_signal[i] = 1
                raw_size[i] = allocation
        df = pd.DataFrame({'signal': raw_signal, 'size': raw_size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=5)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [100, 100, 100, 100, 101, 102, 102, 101],
        'high': [101, 101, 101, 101.2, 103, 102.5, 102, 102],
        'low': [99, 99, 99, 99.5, 100.1, 101.5, 101.3, 100],
        'close': [100, 100, 100, 101, 102.8, 101.8, 101.6, 101],
        'volume': 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1, 1, 1, 1, 1, 2, 2, 2],
        'entry': [False, False, True, False, False, False, False, False],
        'conviction': 0.6
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    # Breakeven arms on bar 3; bar 4 raises the trail to 101.4.
    # ATR expansion on bar 5 must not lower it; bar 6 breaches it.
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert np.isclose(result['size'].iloc[3], 0.67)
    assert (result['size'] > 0).all()
    earlier = data.copy()
    earlier.loc[index[4], 'low'] = 100.0
    stopped = strategy.generate_signals(earlier, indicators, ctx, params).data
    assert stopped['signal'].iloc[5] == 0
    timed = strategy.generate_signals(data, indicators, ctx, GeneratedParams(max_hold=3)).data
    assert timed['signal'].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    short = strategy.indicators(data, params)
    assert not short['entry'].any()
    assert short.index.equals(data.index)


if __name__ == '__main__':
    _self_check()
