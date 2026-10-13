from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendSpreadParams:
    entry_percentile: float = 0.85
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[TrendSpreadParams]):
    strategy_id = 'gen_local_1790898686'

    @classmethod
    def params_type(cls):
        return TrendSpreadParams

    def warmup_bars(self, params: TrendSpreadParams) -> int:
        return 68

    def indicators(self, data: pd.DataFrame, params: TrendSpreadParams) -> pd.DataFrame:
        if not 0.0 < params.entry_percentile <= 1.0 or not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError('Invalid entry percentile or profit target')
        returns = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        net = returns.rolling(5, min_periods=5).sum()
        energy = returns.pow(2).rolling(5, min_periods=5).sum()
        strength = (net.clip(lower=0).pow(2) / (5.0 * energy.replace(0.0, np.nan))).fillna(0.0).where(net.notna())
        percentile = strength.rolling(63, min_periods=63).rank(method='average', pct=True)
        entry = (strength >= 0.6) & (percentile >= params.entry_percentile) & (percentile.shift(1).fillna(0.0) < params.entry_percentile)
        return pd.DataFrame({'returns': returns, 'net': net, 'energy': energy, 'strength': strength, 'percentile': percentile, 'entry': entry.fillna(False)}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TrendSpreadParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_price = 0.0
        pending = False
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_bar = i
                    entry_price = opens[i]
            if entry_bar >= 0:
                # Close-observed targets execute at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or i - entry_bar + 1 >= 2:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = TrendSpreadParams()
    data = pd.DataFrame(100.0, index=pd.date_range('2020-01-01', periods=74), columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = strategy.indicators(data, params)
    assert not indicators['entry'].any()
    indicators.loc[data.index[68], 'entry'] = True
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=68)
    timed = strategy.generate_signals(data, indicators, ctx, params).data
    assert timed['signal'].iloc[68:72].tolist() == [0, 1, 1, 0]
    data.loc[data.index[69], 'close'] = 101.0
    targeted = strategy.generate_signals(data, indicators, ctx, params).data
    assert targeted['signal'].iloc[68:72].tolist() == [0, 1, 0, 0]
    prefix = strategy.generate_signals(data.iloc[:70], indicators.iloc[:70], ctx, params).data
    assert prefix.equals(targeted.iloc[:70])
