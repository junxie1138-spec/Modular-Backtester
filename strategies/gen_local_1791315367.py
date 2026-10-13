from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CommitmentParams:
    percentile_window: int = 126
    streak_quantile: float = 0.85
    min_streak: int = 2
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.2

    def __post_init__(self):
        if self.percentile_window < 2 or self.atr_window < 2 or self.min_streak < 2:
            raise ValueError('Windows and minimum streak must be at least two.')
        if not 0.0 < self.streak_quantile < 1.0:
            raise ValueError('streak_quantile must be between zero and one.')
        if not np.isfinite(self.breakeven_pct) or self.breakeven_pct <= 0:
            raise ValueError('breakeven_pct must be finite and positive.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[CommitmentParams]):
    strategy_id = 'gen_local_1791315367'

    @classmethod
    def params_type(cls):
        return CommitmentParams

    def warmup_bars(self, params: CommitmentParams) -> int:
        return int(max(params.percentile_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CommitmentParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        out['range'] = (data['high'] - data['low']).replace(0.0, np.nan)
        out['body_share'] = (close - data['open']).abs() / out['range']
        out['body_expansion'] = out['body_share'].diff()
        out['close_change'] = close.diff()
        valid = np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        valid &= data[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
        valid &= data['high'].ge(data[['open', 'close']].max(axis=1))
        valid &= data['low'].le(data[['open', 'close']].min(axis=1))
        out['valid'] = valid
        up = (valid & close.gt(data['open']) & out['close_change'].gt(0)
              & out['body_expansion'].gt(0)).fillna(False)
        down = (valid & close.lt(data['open']) & out['close_change'].lt(0)
                & out['body_expansion'].gt(0)).fillna(False)
        out['up_streak'] = up.groupby((~up).cumsum()).cumsum().astype(float)
        out['down_streak'] = down.groupby((~down).cumsum()).cumsum().astype(float)
        out['streak'] = out[['up_streak', 'down_streak']].max(axis=1)
        out['streak_threshold'] = out['streak'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.streak_quantile)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(valid)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        eligible = (valid & out['streak'].ge(params.min_streak)
                    & out['streak'].gt(out['streak_threshold'])
                    & np.isfinite(out['atr']) & out['atr'].gt(0))
        out['candidate'] = np.where(
            eligible, np.where(out['up_streak'].gt(0), 1, -1), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CommitmentParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        valid = indicators['valid'].fillna(False).to_numpy(dtype=bool)
        side = 0
        pending = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        armed = False
        age = 0

        for i in range(len(data)):
            if side:
                if not valid[i]:
                    side = 0
                    pending = False
                    continue
                opening, high, low, closing = prices[i]
                if pending:
                    # Yesterday's decision fills at today's open.
                    entry = opening
                    stop = -params.trail_atr * atr[i - 1]
                    peak = 0.0
                    armed = False
                    age = 0
                    pending = False
                age += 1
                adverse = low - entry if side == 1 else entry - high
                if adverse <= stop:
                    # Stop observations submit an exit for the next open.
                    side = 0
                    continue
                favorable = high - entry if side == 1 else entry - low
                peak = max(peak, favorable)
                if peak >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed profit-space stop only rises; short price stops fall.
                    stop = max(stop, 0.0)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                closing_profit = side * (closing - entry)
                # The close follows the day's favorable extreme, so it can
                # safely confirm a breach of the newly tightened stop.
                if closing_profit <= stop or age >= 2:
                    side = 0
                    continue
                decisions[i] = side
            elif valid[i] and candidates[i] != 0:
                side = int(candidates[i])
                pending = True
                decisions[i] = side

        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = CommitmentParams(trail_atr=1.0, breakeven_pct=0.01)
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 100.0, 100.0],
        'high': [100.5, 102.0, 100.5, 100.5],
        'low': [99.5, 99.5, 99.5, 99.5],
        'close': [100.0, 100.5, 100.0, 100.0],
        'volume': [1000.0] * 4,
    }, index=index)
    ind = pd.DataFrame({
        'atr': [1.0] * 4, 'valid': [True] * 4,
        'candidate': [1, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    mirrored = data.copy()
    mirrored['high'] = 200.0 - data['low']
    mirrored['low'] = 200.0 - data['high']
    mirrored['close'] = 200.0 - data['close']
    ind['candidate'] = [-1, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, mirrored, ind, None, params).data
    assert result['signal'].tolist() == [0, -1, 0, 0]
    data.loc[index[1], ['high', 'close']] = [100.5, 100.2]
    ind['candidate'] = [1, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
