from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_bars: int = 3
    noise_bars: int = 8
    snr_threshold: float = 1.5
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791259318'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.streak_bars, params.noise_bars, params.atr_bars) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.streak_bars, params.noise_bars, params.atr_bars) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.snr_threshold, params.breakeven_pct, params.trail_atr)):
            raise ValueError('Thresholds and ATR multiplier must be finite and positive.')

        out = pd.DataFrame(index=data.index)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['midpoint_change'] = out['midpoint'].diff()
        out['close_change'] = data['close'].diff()
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars,
                                               min_periods=params.atr_bars).mean()
        out['displacement'] = out['close_change'].rolling(
            params.noise_bars, min_periods=params.noise_bars).sum()
        out['noise'] = np.sqrt(out['close_change'].pow(2).rolling(
            params.noise_bars, min_periods=params.noise_bars).sum())
        out['snr'] = out['displacement'].div(out['noise'].where(out['noise'] > 0))

        # Saturated streak counts require only streak_bars + 1 observations.
        up_alive = pd.Series(True, index=data.index)
        down_alive = pd.Series(True, index=data.index)
        up_count = pd.Series(0, index=data.index, dtype=int)
        down_count = pd.Series(0, index=data.index, dtype=int)
        for lag in range(params.streak_bars):
            movement = out['midpoint_change'].shift(lag)
            up_alive = up_alive & movement.gt(0)
            down_alive = down_alive & movement.lt(0)
            up_count = up_count + up_alive.astype(int)
            down_count = down_count + down_alive.astype(int)
        out['up_streak'] = up_count
        out['down_streak'] = down_count
        ready = (out['atr'].gt(0) & np.isfinite(out['atr'])
                 & np.isfinite(out['snr']))
        long_setup = ready & out['up_streak'].eq(params.streak_bars) & out['snr'].ge(params.snr_threshold)
        short_setup = ready & out['down_streak'].eq(params.streak_bars) & out['snr'].le(-params.snr_threshold)
        out['setup'] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        out['entry_event'] = out['setup'].where(
            out['setup'].ne(out['setup'].shift(1).fillna(0)), 0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        events = indicators['entry_event'].to_numpy(dtype=int)
        position = 0
        pending = False
        entry = 0.0
        stop = 0.0
        watermark = 0.0
        armed = False
        age = 0
        warmup = GeneratedStrategy.warmup_bars(params)

        for i, (opening, high, low, close) in enumerate(prices):
            valid = (np.isfinite([opening, high, low, close]).all()
                     and min(opening, high, low, close) > 0
                     and low <= min(opening, close)
                     and high >= max(opening, close))
            if not valid:
                position = 0
                pending = False
                continue

            if pending:
                # The prior close's decision first fills at this bar's open.
                entry = opening
                watermark = position * entry
                stop = watermark - params.trail_atr * atr[i - 1]
                armed = False
                age = 0
                pending = False

            if position:
                age += 1
                # Signed prices let both long and short stops ratchet upward.
                adverse = low if position == 1 else -high
                favorable_close = position * close
                if adverse <= stop or age >= params.hold_bars or not np.isfinite(atr[i]):
                    position = 0
                    continue
                watermark = max(watermark, favorable_close)
                if position * (close / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Close-confirmed stop changes become active on the next bar.
                    stop = max(stop, position * entry,
                               watermark - params.trail_atr * atr[i])
                decisions[i] = position
            elif i >= warmup - 1 and events[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(events[i])
                pending = True
                decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params(streak_bars=2, noise_bars=3, atr_bars=2, snr_threshold=1.0)
    for direction in (1, -1):
        close = 100.0 + direction * np.arange(12)
        data = pd.DataFrame({
            'open': close - direction * 0.1,
            'high': close + 0.2,
            'low': close - 0.2,
            'close': close,
            'volume': 1000000.0
        }, index=pd.date_range('2020-01-01', periods=12))
        features = GeneratedStrategy.indicators(data, params)
        result = GeneratedStrategy.generate_signals(data, features, None, params).data
        assert result.index.equals(data.index)
        assert result['signal'].iloc[:4].eq(0).all()
        assert result['signal'].iloc[4:8].eq(direction).all()
        assert result['signal'].iloc[8:].eq(0).all()
        assert result['size'].gt(0).all()
        if direction == 1:
            data.iloc[6, data.columns.get_loc('low')] = 100.0
        else:
            data.iloc[6, data.columns.get_loc('high')] = 100.0
        features = GeneratedStrategy.indicators(data, params)
        stopped = GeneratedStrategy.generate_signals(data, features, None, params).data
        assert stopped['signal'].iloc[6] == direction
        assert stopped['signal'].iloc[7] == 0
