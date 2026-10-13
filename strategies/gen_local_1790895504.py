from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_window: int = 40
    depth_threshold: float = 0.035
    dwell_bars: int = 3
    rearm_fraction: float = 0.5
    atr_window: int = 14
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = 'gen_local_1790895504'

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(200, params.depth_window + params.dwell_bars + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if min(params.depth_window, params.dwell_bars, params.atr_window, params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0 < params.depth_threshold < 1 and 0 < params.rearm_fraction < 1):
            raise ValueError('Depth threshold and rearm fraction must lie between zero and one.')
        if min(params.initial_stop_atr, params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError('Stop and activation parameters must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out['trough'] = close.rolling(params.depth_window, min_periods=params.depth_window).min()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['rebound'] = close / out['trough'].where(out['trough'] > 0) - 1.0
        out['previous_drawdown'] = out['drawdown'].shift(1)
        out['previous_rebound'] = out['rebound'].shift(1)
        out['drawdown_dwell'] = out['drawdown'].shift(1).rolling(params.dwell_bars, min_periods=params.dwell_bars).min()
        out['rebound_dwell'] = out['rebound'].shift(1).rolling(params.dwell_bars, min_periods=params.dwell_bars).min()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ready'] = np.isfinite(out).all(axis=1) & np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1) & (close > 0) & (data['open'] > 0) & (out['atr'] > 0)
        out['long_entry'] = out['ready'] & (close > out['ma200']) & (out['drawdown_dwell'] >= params.depth_threshold) & (out['drawdown'] < out['previous_drawdown'])
        out['short_entry'] = out['ready'] & (close < out['ma200']) & (out['rebound_dwell'] >= params.depth_threshold) & (out['rebound'] < out['previous_rebound'])
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        drawdowns = indicators['drawdown'].to_numpy(dtype=float)
        rebounds = indicators['rebound'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        long_entries = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        short_entries = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        long_armed = True
        short_armed = True
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        stop_offset = -np.inf
        activated = False
        held = 0
        rearm_depth = params.depth_threshold * params.rearm_fraction

        for i in range(count):
            if np.isfinite(drawdowns[i]) and drawdowns[i] < rearm_depth:
                long_armed = True
            if np.isfinite(rebounds[i]) and rebounds[i] < rearm_depth:
                short_armed = True

            if position:
                # A decision made yesterday fills at today's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    stop_offset = -params.initial_stop_atr * entry_atr
                held += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    signals[i] = position
                    continue
                profit_offset = position * (closes[i] - entry_price)
                exit_now = profit_offset <= stop_offset or held >= params.max_hold_bars
                if not exit_now:
                    if profit_offset >= entry_price * params.breakeven_pct:
                        activated = True
                    if activated:
                        # Signed profit coordinates make the stop ratchet upward for both directions.
                        stop_offset = max(stop_offset, 0.0)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop_offset = max(stop_offset, profit_offset - params.trail_atr * atrs[i])
                        exit_now = profit_offset <= stop_offset
                if exit_now:
                    position = 0
                    entry_price = np.nan
                    activated = False
                    held = 0
                signals[i] = position
                continue

            if ready[i]:
                if long_armed and long_entries[i]:
                    position = 1
                    long_armed = False
                elif short_armed and short_entries[i]:
                    position = -1
                    short_armed = False
                if position:
                    entry_atr = atrs[i]
                    entry_price = np.nan
                    activated = False
                    held = 0
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': np.full(count, 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = DrawdownParams()
    for direction in (1, -1):
        index = pd.date_range('2020-01-01', periods=5)
        close = np.array([100.0, 100.0 + 2.0 * direction, 100.0 - direction, 100.0, 100.0])
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({'atr': 1.0, 'drawdown': 0.05, 'rebound': 0.05, 'ready': True, 'long_entry': False, 'short_entry': False}, index=index)
        indicators.loc[index[0], 'long_entry' if direction == 1 else 'short_entry'] = True
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]
        assert (result['size'] > 0).all()
