from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 6
    loss_threshold: float = 0.025
    profit_target: float = 0.012
    hold_bars: int = 4

    def __post_init__(self):
        if not 4 <= self.lookback <= 8:
            raise ValueError('lookback must be between 4 and 8')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between 3 and 5')
        if not np.isfinite(self.loss_threshold) or self.loss_threshold <= 0:
            raise ValueError('loss_threshold must be finite and positive')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791198866'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close'].astype(float).where(data['close'] > 0)
        returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        losses = (-returns).clip(lower=0)
        prior_losses = losses.shift(1)
        window = params.lookback
        loss_mass = prior_losses.rolling(window, min_periods=window).sum()
        largest_loss = prior_losses.rolling(window, min_periods=window).max()
        negative = returns.lt(0).astype(float).where(returns.notna())
        loss_count = negative.shift(1).rolling(window, min_periods=window).sum()
        prior_peak = close.shift(1).rolling(window, min_periods=window).max()
        drawdown = close / prior_peak - 1.0
        entry = (
            (loss_mass >= params.loss_threshold)
            & (loss_count >= (window + 1) // 2)
            & (largest_loss <= 0.60 * loss_mass)
            & (returns > 0)
            & (returns.shift(1) <= 0)
            & (returns >= 0.80 * largest_loss)
            & (returns <= 0.75 * loss_mass)
            & (drawdown < 0)
        ).fillna(False)
        return pd.DataFrame({
            'log_return': returns.fillna(0.0),
            'loss_packet': losses.fillna(0.0),
            'prior_loss_mass': loss_mass.fillna(0.0),
            'prior_largest_loss': largest_loss.fillna(0.0),
            'prior_loss_count': loss_count.fillna(0.0),
            'prior_peak': prior_peak.fillna(0.0),
            'drawdown': drawdown.fillna(0.0),
            'entry': entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        age = 0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and signal[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                active = True
                entry_price = opens[i]
                age = 0
            if active:
                age += 1
                reached_target = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if reached_target or age >= params.hold_bars:
                    active = False
                    # The zero decision exits at the following open.
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0, 'high': 103.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True] + [False] * 6}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], 'close'] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert GeneratedStrategy.warmup_bars(None, RecoveryParams(lookback=8)) == 10
    flat = data.copy()
    flat['close'] = 100.0
    derived = GeneratedStrategy.indicators(None, flat, params)
    assert not derived.isna().any().any()
    assert not derived['entry'].any()


if __name__ == '__main__':
    _self_check()
