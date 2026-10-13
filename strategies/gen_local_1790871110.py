from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRelaxationParams:
    fast_atr: int = 5
    slow_atr: int = 20
    peak_window: int = 40
    drawdown_atr: float = 2.0
    flow_window: int = 5
    flow_threshold: float = 0.10
    profit_target: float = 0.04
    max_hold: int = 20
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[ShockRelaxationParams]):
    strategy_id = 'gen_local_1790871110'

    @classmethod
    def params_type(cls):
        return ShockRelaxationParams

    def warmup_bars(self, params: ShockRelaxationParams) -> int:
        # Previous close for TR, plus previous ATR ratio for the crossing.
        return int(max(params.peak_window, params.slow_atr + 2,
                       params.fast_atr + 2, params.flow_window))

    def indicators(self, data: pd.DataFrame,
                   params: ShockRelaxationParams) -> pd.DataFrame:
        if not (1 <= params.fast_atr < params.slow_atr
                and params.peak_window >= 1 and params.flow_window >= 1):
            raise ValueError('Require positive windows and fast_atr < slow_atr')
        if not (np.isfinite(params.drawdown_atr) and params.drawdown_atr > 0
                and np.isfinite(params.flow_threshold)
                and 0 <= params.flow_threshold < 1):
            raise ValueError('Invalid drawdown or money-flow threshold')

        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['range', 'high_gap', 'low_gap']].max(axis=1)
        out['fast_atr'] = out['true_range'].rolling(params.fast_atr).mean()
        out['slow_atr'] = out['true_range'].rolling(params.slow_atr).mean()
        out['atr_ratio'] = out['fast_atr'].div(out['slow_atr'].replace(0, np.nan))
        out['previous_atr_ratio'] = out['atr_ratio'].shift(1)
        out['peak_close'] = data['close'].rolling(params.peak_window).max()
        out['drawdown_atr'] = (out['peak_close'] - data['close']).div(
            out['slow_atr'].replace(0, np.nan))

        # Zero-range bars contribute neutral closing-location pressure.
        out['close_location'] = (
            (2 * data['close'] - data['high'] - data['low'])
            .div(out['range'].replace(0, np.nan)).clip(-1, 1).fillna(0.0)
        )
        out['signed_volume'] = out['close_location'] * data['volume']
        out['flow_sum'] = out['signed_volume'].rolling(params.flow_window).sum()
        out['volume_sum'] = data['volume'].rolling(params.flow_window).sum()
        out['money_flow'] = out['flow_sum'].div(out['volume_sum'].replace(0, np.nan))
        out['volatility_agrees'] = (
            out['previous_atr_ratio'].ge(1.0)
            & out['atr_ratio'].lt(1.0)
            & out['drawdown_atr'].ge(params.drawdown_atr)
        )
        out['flow_agrees'] = out['money_flow'].gt(params.flow_threshold)
        out['entry'] = (out['volatility_agrees'] & out['flow_agrees']).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: ShockRelaxationParams) -> SignalFrame:
        if not (params.max_hold >= 1
                and np.isfinite(params.profit_target) and params.profit_target > 0
                and np.isfinite(params.size) and params.size > 0):
            raise ValueError('Holding period, target, and size must be positive')

        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw_signal = np.zeros(len(data), dtype=int)
        decision_bar = -1
        entry_price = np.nan
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if decision_bar >= 0:
                # Observe the entry open only when that bar arrives.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        raise ValueError('Entry open must be finite and positive')
                held_sessions = i - decision_bar
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held_sessions >= params.max_hold:
                    decision_bar = -1
                    entry_price = np.nan
                else:
                    raw_signal[i] = 1
            elif i >= warmup and entries[i]:
                decision_bar = i
                raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw_signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShockRelaxationParams(fast_atr=1, slow_atr=2, peak_window=2,
                                   flow_window=1, max_hold=3)
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d',
                          warmup_bars=strategy.warmup_bars(params))
    indicators = pd.DataFrame({'entry': False}, index=index)
    indicators.loc[index[4], 'entry'] = True

    timed = strategy.generate_signals(data, indicators, ctx, params).data
    assert timed['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[6], 'close'] = 104.1
    targeted = strategy.generate_signals(data, indicators, ctx, params).data
    assert targeted['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    assert targeted['size'].gt(0).all()

    for length in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:length],
                                            indicators.iloc[:length], ctx, params).data
        assert prefix.equals(targeted.iloc[:length])

    # A gap changes the target basis to the entry open.
    data.loc[index[5], 'open'] = 110.0
    gapped = strategy.generate_signals(data, indicators, ctx, params).data
    assert gapped['signal'].equals(timed['signal'])
    derived = strategy.indicators(data, params)
    assert derived['entry'].equals(derived['volatility_agrees'] & derived['flow_agrees'])
    assert not derived['entry'].iloc[:strategy.warmup_bars(params)].any()


if __name__ == '__main__':
    _self_check()
