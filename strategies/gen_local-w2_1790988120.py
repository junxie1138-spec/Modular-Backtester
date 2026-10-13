from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthPropagationParams:
    atr_window: int = 14
    min_streak: int = 3
    entry_sessions: int = 6
    hold_bars: int = 18
    stop_atr: float = 2.5
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[MonthPropagationParams]):
    strategy_id = 'gen_local-w2_1790988120'

    @classmethod
    def params_type(cls) -> type[MonthPropagationParams]:
        return MonthPropagationParams

    def warmup_bars(self, params: MonthPropagationParams) -> int:
        return max(1, int(params.atr_window)) + 1

    def indicators(self, data: pd.DataFrame, params: MonthPropagationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous = close.shift(1)
        change = close.diff()
        month = pd.Series(data.index.to_period('M'), index=data.index)
        new_month = month.ne(month.shift(1))
        session = month.groupby(month, sort=False).cumcount() + 1
        up = change.gt(0) & change.notna()
        down = change.lt(0) & change.notna()
        up_groups = (new_month | ~up).cumsum()
        down_groups = (new_month | ~down).cumsum()
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        window = max(1, int(params.atr_window))
        atr = true_range.rolling(window, min_periods=window).mean()
        month_anchor = previous.where(new_month).groupby(month, sort=False).ffill()
        displacement = (close - month_anchor) / atr.where(atr.gt(0))
        up_streak = up.astype(int).groupby(up_groups, sort=False).cumsum()
        down_streak = down.astype(int).groupby(down_groups, sort=False).cumsum()
        streak = pd.concat([up_streak, down_streak], axis=1).max(axis=1)
        lower = max(0.01, min(1.0, float(params.min_size)))
        upper = max(lower, min(1.0, float(params.max_size)))
        strength = 0.25 + 0.10 * streak + 0.05 * displacement.abs().fillna(0.0)
        out['atr'] = atr
        out['month_code'] = data.index.year * 12 + data.index.month
        out['month_session'] = session
        out['up_streak'] = up_streak
        out['down_streak'] = down_streak
        out['month_displacement'] = displacement.replace([np.inf, -np.inf], np.nan)
        out['conviction_size'] = strength.clip(lower=lower, upper=upper).fillna(lower)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MonthPropagationParams) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        lower = max(0.01, min(1.0, float(params.min_size)))
        raw_size = np.full(n, lower, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        months = indicators['month_code'].to_numpy(dtype=int)
        sessions = indicators['month_session'].to_numpy(dtype=int)
        up = indicators['up_streak'].to_numpy(dtype=int)
        down = indicators['down_streak'].to_numpy(dtype=int)
        displacement = indicators['month_displacement'].to_numpy(dtype=float)
        sizes = indicators['conviction_size'].to_numpy(dtype=float)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        frozen_atr = np.nan
        position_size = lower
        used_month = -1
        minimum = max(1, int(params.min_streak))
        entry_sessions = max(minimum, int(params.entry_sessions))
        holding = max(1, int(params.hold_bars))
        stop_multiple = max(0.01, float(params.stop_atr))

        for i in range(n):
            exited = False
            if direction != 0 and not np.isfinite(entry_price):
                # Yesterday's decision fills at today's open.
                if np.isfinite(opening[i]) and opening[i] > 0:
                    entry_price = opening[i]
                    entry_bar = i
                else:
                    direction = 0
                    exited = True

            if direction != 0 and np.isfinite(close[i]):
                # The stop distance is frozen using the decision bar's ATR.
                stopped = direction * (close[i] - entry_price) < -stop_multiple * frozen_atr
                expired = i - entry_bar + 1 >= holding
                if stopped or expired:
                    direction = 0
                    entry_price = np.nan
                    exited = True

            if direction == 0 and not exited and months[i] != used_month:
                ready = (np.isfinite(atr[i]) and atr[i] > 0
                         and np.isfinite(displacement[i])
                         and np.isfinite(close[i]) and close[i] > 0)
                if ready and sessions[i] <= entry_sessions:
                    candidate = 0
                    if up[i] >= minimum and displacement[i] > 0:
                        candidate = 1
                    elif down[i] >= minimum and displacement[i] < 0:
                        candidate = -1
                    if candidate != 0:
                        direction = candidate
                        entry_price = np.nan
                        frozen_atr = atr[i]
                        position_size = sizes[i]
                        used_month = months[i]

            raw_signal[i] = direction
            raw_size[i] = position_size if direction != 0 else lower

        df = pd.DataFrame({'signal': raw_signal, 'size': raw_size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(lower).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = MonthPropagationParams(atr_window=3, stop_atr=2.0)
    index = pd.bdate_range('2020-01-06', periods=12)
    for side in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0,
                             'close': 100.0, 'volume': 1000.0}, index=index)
        data.loc[index[6], 'close'] = 100.0 - side * 3.0
        indicators = strategy.indicators(data, params)
        indicators['atr'] = 1.0
        indicators['up_streak'] = 0
        indicators['down_streak'] = 0
        indicators['month_displacement'] = 0.0
        indicators['conviction_size'] = 0.7
        indicators.loc[index[4], 'up_streak' if side == 1 else 'down_streak'] = 3
        indicators.loc[index[4], 'month_displacement'] = float(side)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].iloc[4] == 0
        assert result['signal'].iloc[5] == side
        assert result['signal'].iloc[6] == side
        assert result['signal'].iloc[7] == 0
        assert result['size'].iloc[5] == 0.7
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
