from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    tail_quantile: float = 0.90
    atr_period: int = 14
    holding_bars: int = 4
    breakeven_pct: float = 0.008
    initial_stop_atr: float = 1.5
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791219490'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError('tail_quantile must be between 0.5 and 1.')
        if not 3 <= params.holding_bars <= 5:
            raise ValueError('holding_bars must be between three and five.')
        if not all(np.isfinite(x) and x > 0 for x in (
            params.breakeven_pct, params.initial_stop_atr, params.trail_atr
        )):
            raise ValueError('Stop parameters must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1).replace(0.0, np.nan)
        out['return'] = data['close'] / previous - 1.0
        out['gap'] = data['open'] / previous - 1.0
        for column in ('return', 'gap'):
            history = out[column].shift(1).rolling(params.lookback, min_periods=params.lookback)
            out[column + '_upper'] = history.quantile(params.tail_quantile)
            out[column + '_lower'] = history.quantile(1.0 - params.tail_quantile)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        ready = out.notna().all(axis=1) & (out['atr'] > 0)
        long_entry = (
            ready & (out['return'] > 0) & (out['gap'] > 0)
            & (out['return'] > out['return_upper'])
            & (out['gap'] > out['gap_upper'])
        )
        short_entry = (
            ready & (out['return'] < 0) & (out['gap'] < 0)
            & (out['return'] < out['return_lower'])
            & (out['gap'] < out['gap_lower'])
        )
        out['entry_direction'] = long_entry.astype(int) - short_entry.astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = pending = age = 0
        entry = best = stop = 0.0
        armed = False
        for i in range(len(data)):
            # Yesterday's decision fills at today's open; only yesterday's ATR is used.
            if pending:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = pending
                    entry = opens[i]
                    best = entry
                    stop = entry - position * params.initial_stop_atr * atr[i - 1]
                    age = 0
                    armed = False
                pending = 0
            if position:
                age += 1
                close = closes[i]
                if not np.isfinite(close) or close <= 0:
                    position = 0
                else:
                    # Close-evaluated stops schedule an exit at the following open.
                    stopped = close <= stop if position == 1 else close >= stop
                    if position == 1:
                        best = max(best, close)
                        armed = armed or best >= entry * (1.0 + params.breakeven_pct)
                        if armed:
                            stop = max(stop, entry)
                            if np.isfinite(atr[i]) and atr[i] > 0:
                                stop = max(stop, best - params.trail_atr * atr[i])
                        stopped = stopped or close <= stop
                    else:
                        best = min(best, close)
                        armed = armed or best <= entry * (1.0 - params.breakeven_pct)
                        if armed:
                            stop = min(stop, entry)
                            if np.isfinite(atr[i]) and atr[i] > 0:
                                stop = min(stop, best + params.trail_atr * atr[i])
                        stopped = stopped or close >= stop
                    if stopped or age >= params.holding_bars:
                        position = 0
                decisions[i] = position
                # An exit decision stays flat; no same-close reversal.
                continue
            if entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = int(entries[i])
                decisions[i] = pending
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params()
    for direction in (1, -1):
        close = np.full(80, 100.0)
        opening = close.copy()
        close[60:64] = [105.0, 106.0, 108.0, 105.0]
        opening[60:64] = [104.0, 106.0, 106.0, 108.0]
        if direction == -1:
            close, opening = 200.0 - close, 200.0 - opening
        data = pd.DataFrame({
            'open': opening, 'close': close,
            'high': np.maximum(opening, close) + 0.5,
            'low': np.minimum(opening, close) - 0.5,
            'volume': 1000000.0,
        }, index=pd.date_range('2020-01-01', periods=80))
        result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
        assert result['signal'].iloc[:61].eq(0).all()
        assert result['signal'].iloc[61:64].eq(direction).all()
        assert result['signal'].iloc[64] == 0
        assert result['signal'].isin([-1, 0, 1]).all() and result['size'].gt(0).all()
        prefix = data.iloc[:64]
        earlier = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
        assert earlier.equals(result.iloc[:64])
