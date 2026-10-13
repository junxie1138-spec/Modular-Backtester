from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectionParams:
    rejection_window: int = 10
    drift_window: int = 15
    rank_window: int = 126
    rejection_percentile: float = 0.85
    drift_percentile: float = 0.20
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RejectionParams]):
    strategy_id = 'gen_local_1791245963'

    @classmethod
    def params_type(cls) -> type[RejectionParams]:
        return RejectionParams

    def warmup_bars(self, params: RejectionParams) -> int:
        return int(max(params.rejection_window + params.rank_window - 1,
                       params.drift_window + params.rank_window))

    def indicators(self, data: pd.DataFrame, params: RejectionParams) -> pd.DataFrame:
        if min(params.rejection_window, params.drift_window, params.rank_window,
               params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive')
        if not (0 < params.rejection_percentile <= 1 and
                0 < params.drift_percentile <= 1 and
                np.isfinite(params.profit_target) and params.profit_target > 0):
            raise ValueError('Invalid percentile or profit target')
        out = pd.DataFrame(index=data.index)
        body_low = data[['open', 'close']].min(axis=1, skipna=False)
        body_high = data[['open', 'close']].max(axis=1, skipna=False)
        out['lower_wick'] = (body_low - data['low']).clip(lower=0)
        out['upper_wick'] = (data['high'] - body_high).clip(lower=0)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        out['tail_balance'] = out['lower_wick'] - out['upper_wick']
        out['tail_sum'] = out['tail_balance'].rolling(params.rejection_window).sum()
        out['range_sum'] = out['range'].rolling(params.rejection_window).sum()
        out['rejection'] = out['tail_sum'] / out['range_sum'].replace(0, np.nan)
        out['drift'] = data['close'] / data['close'].shift(params.drift_window).replace(0, np.nan) - 1.0
        out['rejection_rank'] = out['rejection'].rolling(params.rank_window).rank(pct=True)
        out['drift_rank'] = out['drift'].rolling(params.rank_window).rank(pct=True)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            (out['rejection_rank'] >= params.rejection_percentile)
            & (out['drift_rank'] <= params.drift_percentile)
            & (out['drift'] < 0)
            & (out['rejection'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RejectionParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    entry_bar = i
                    active = True
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 100.0, 'low': 100.0,
        'close': 100.0, 'volume': 1000.0
    }, index=index)
    indicators = pd.DataFrame({'entry': True}, index=index)
    timed = strategy.generate_signals(
        data, indicators, None,
        RejectionParams(max_hold_bars=2, profit_target=0.50)
    ).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    indicators['entry'] = [True, False, False, False, False, False]
    data.loc[index[1], 'open'] = 102.0
    data.loc[index[1:], 'close'] = [104.0, 106.0, 106.0, 106.0, 106.0]
    targeted = strategy.generate_signals(
        data, indicators, None,
        RejectionParams(max_hold_bars=18, profit_target=0.03)
    ).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0]
