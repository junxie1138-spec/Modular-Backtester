from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    coupling_threshold: float = 0.30
    profit_target: float = 0.035
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791273411'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # Two four-pair blocks require ten closing prices.
        return 10

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0 < params.coupling_threshold < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        out['return'] = (data['close'] / data['close'].shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        out['lag_return'] = out['return'].shift(1)
        out['pair_product'] = out['return'] * out['lag_return']
        out['pair_energy'] = (out['return'] ** 2 + out['lag_return'] ** 2) / 2
        out['coupling'] = out['pair_product'].rolling(4).sum() / out['pair_energy'].rolling(4).sum().replace(0, np.nan)
        out['previous_coupling'] = out['coupling'].shift(4)
        out['displacement'] = out['return'].rolling(4).sum()
        out['setup'] = (out['coupling'] > params.coupling_threshold) & (out['previous_coupling'] < -params.coupling_threshold) & (out['displacement'] > 0) & (out['return'] > 0)
        out['entry'] = out['setup'] & ~out['setup'].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # Read the entry open only after the decision bar has passed.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                elif (np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)) or i - entry_bar + 1 >= params.max_hold:
                    active = False
            elif entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = int(active)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=16)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry': np.arange(16) == 9}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=10)
    result = strategy.generate_signals(data, entries, ctx, Params(profit_target=10.0, max_hold=3)).data
    assert result['signal'].tolist() == [0] * 10 + [1] * 3 + [0] * 3
    data.loc[index[10], 'open'] = 120.0
    data.loc[index[11], 'close'] = 125.0
    result = strategy.generate_signals(data, entries, ctx, Params()).data
    assert result['signal'].tolist() == [0] * 10 + [1] * 2 + [0] * 4
    params = Params()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:12], params)
    pd.testing.assert_frame_equal(full.iloc[:12], prefix)
    assert not full.isna().any().any()
    assert strategy.warmup_bars(params) == 10


if __name__ == '__main__':
    _self_check()
