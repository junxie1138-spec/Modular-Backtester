from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapResponseParams:
    response_window: int = 30
    min_gap_days: int = 4
    response_spread: float = 0.002
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapResponseParams]):
    strategy_id = 'gen_local-w2_1790965977'

    @classmethod
    def params_type(cls):
        return GapResponseParams

    def warmup_bars(self, params: GapResponseParams) -> int:
        return int(max(200, params.response_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapResponseParams) -> pd.DataFrame:
        if not (params.response_window >= 2 and params.atr_window >= 2
                and 1 <= params.min_gap_days <= params.response_window // 2
                and params.response_spread >= 0 and params.breakeven_pct > 0
                and params.trail_atr > 0 and params.hold_bars >= 1):
            raise ValueError('Invalid gap-response parameters')
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1).replace(0.0, np.nan)
        out['return'] = (data['close'] / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        out['gap'] = (data['open'] / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        valid = out['return'].notna() & out['gap'].notna()
        down = valid & out['gap'].lt(0)
        up = valid & out['gap'].gt(0)
        window = params.response_window
        out['down_count'] = down.astype(float).rolling(window, min_periods=window).sum()
        out['up_count'] = up.astype(float).rolling(window, min_periods=window).sum()
        out['down_response'] = out['return'].where(down, 0.0).rolling(window, min_periods=window).sum() / out['down_count'].replace(0.0, np.nan)
        out['up_response'] = out['return'].where(up, 0.0).rolling(window, min_periods=window).sum() / out['up_count'].replace(0.0, np.nan)
        out['response_difference'] = out['down_response'] - out['up_response']
        out['mean_return'] = out['return'].rolling(window, min_periods=window).mean()
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['entry'] = (
            out['response_difference'].gt(params.response_spread)
            & out['response_difference'].shift(1).le(params.response_spread)
            & out['down_count'].ge(params.min_gap_days)
            & out['up_count'].ge(params.min_gap_days)
            & out['down_response'].gt(0)
            & out['mean_return'].gt(0)
            & out['return'].gt(0)
            & data['close'].gt(out['ma200'])
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapResponseParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                # The previous close's decision fills at this session's open.
                entry = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                peak = entry
                age = 0
                active, pending, armed = True, False, False
            if active:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]):
                    if age >= params.hold_bars:
                        active = False
                    else:
                        signal[i] = 1
                    continue
                # Stops are evaluated at closes; exits execute next session.
                peak = max(peak, close[i])
                if close[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and i + 1 < len(data):
                pending = True
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({
        'open': 100.0,
        'high': 105.0,
        'low': 99.0,
        'close': [100.0, 101.0, 104.0, 103.0, 102.0, 101.0, 102.0, 103.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0,
        'entry': [True, False, False, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapResponseParams(trail_atr=2.0)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data['close'] = 100.0
    params.hold_bars = 3
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert not strategy.indicators(data, params)['entry'].any()
