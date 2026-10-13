from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakRegimeParams:
    regime_window: int = 60
    pullback_days: int = 3
    regime_gap: float = 0.06
    atr_window: int = 14
    stop_atr: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.regime_window, self.pullback_days, self.atr_window, self.holding_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be positive and finite.')
        if not np.isfinite(self.regime_gap) or not 0 < self.regime_gap <= 1:
            raise ValueError('regime_gap must be in (0, 1].')


class GeneratedStrategy(BaseStrategy[StreakRegimeParams]):
    strategy_id = 'gen_local_1791303564'

    @classmethod
    def params_type(cls):
        return StreakRegimeParams

    def warmup_bars(self, params: StreakRegimeParams) -> int:
        cap = max(3, params.pullback_days)
        return int(max(params.regime_window + cap + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StreakRegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        change = close.diff()
        up = change.gt(0) & change.notna()
        down = change.lt(0) & change.notna()
        cap = max(3, params.pullback_days)
        out['up_streak'] = up.astype(int).groupby((~up).cumsum()).cumsum().clip(upper=cap)
        out['down_streak'] = down.astype(int).groupby((~down).cumsum()).cumsum().clip(upper=cap)
        out['up_persistence'] = out['up_streak'].ge(3).astype(float).rolling(params.regime_window, min_periods=params.regime_window).mean()
        out['down_persistence'] = out['down_streak'].ge(3).astype(float).rolling(params.regime_window, min_periods=params.regime_window).mean()
        out['persistence_gap'] = out['up_persistence'] - out['down_persistence']
        out['trend_regime'] = out['persistence_gap'].gt(params.regime_gap)
        out['balanced_regime'] = out['persistence_gap'].abs().le(params.regime_gap)
        out['recent_down_streak'] = out['down_streak'].shift(1).rolling(2, min_periods=2).max()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        recovery = out['balanced_regime'] & out['recent_down_streak'].ge(params.pullback_days)
        out['candidate'] = up & (out['trend_regime'] | recovery) & out['atr'].gt(0)
        out['candidate'] &= out['persistence_gap'].notna() & out['atr'].notna()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakRegimeParams) -> SignalFrame:
        candidate = indicators['candidate'].fillna(False).astype(bool)
        confirmed = (candidate & candidate.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        stop_price = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    stop_price = opens[i] - params.stop_atr * entry_atr
            if active:
                stop_hit = np.isfinite(closes[i]) and closes[i] < stop_price
                time_exit = i - entry_bar + 1 >= params.holding_bars
                if stop_hit or time_exit or not np.isfinite(closes[i]):
                    active = False
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                entry_atr = atr[i]
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100, 100, 100, 100, 104, 96, 93],
        'close': [100, 100, 100, 104, 96, 93, 93],
    }, index=index)
    indicators = pd.DataFrame({
        'candidate': [True, True, False, False, False, False, False],
        'atr': [2.0] * 7,
    }, index=index)
    params = StreakRegimeParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0]
    indicators['candidate'] = [True, False, False, False, False, False, False]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].eq(0).all()
    indicators['candidate'] = [True, True, False, False, False, False, False]
    params.holding_bars = 3
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0]


if __name__ == '__main__':
    _self_check()
