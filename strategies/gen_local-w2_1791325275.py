from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    pattern_window: int = 18
    transition_bars: int = 5
    peak_window: int = 63
    min_drawdown: float = 0.025
    max_drawdown: float = 0.12
    disorder_floor: float = 0.80
    entropy_drop: float = 0.08
    profit_target: float = 0.04
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791325275'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window,
                       params.pattern_window + 2 + params.transition_bars))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if (params.pattern_window < 2 or params.transition_bars < 1
                or params.peak_window < 3 or params.time_stop < 1
                or not 0 < params.min_drawdown < params.max_drawdown < 1
                or not 0 < params.disorder_floor <= 1
                or not 0 < params.entropy_drop < 1
                or not 0 < params.profit_target < 1):
            raise ValueError('Invalid recovery parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & data['close'].gt(0))
        out['close_lag1'] = close.shift(1)
        out['close_lag2'] = close.shift(2)
        a, b = out['close_lag2'], out['close_lag1']
        valid = a.notna() & b.notna() & close.notna()
        # Equal values are ordered by their chronological position.
        out['ordinal_code'] = (a.le(b).astype(int)
                               + 2 * a.le(close).astype(int)
                               + 4 * b.le(close).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            frequency = out['ordinal_code'].eq(code).astype(float).where(valid)
            probability = frequency.rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            out[f'pattern_{code}_frequency'] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out['entropy'] = entropy / np.log(6.0)
        out['previous_entropy'] = out['entropy'].shift(params.transition_bars)
        out['entropy_contraction'] = out['previous_entropy'] - out['entropy']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out['drawdown'] = 1.0 - close / out['peak']
        out['entry'] = (
            close.gt(out['ma200'])
            & out['drawdown'].between(params.min_drawdown, params.max_drawdown)
            & out['previous_entropy'].ge(params.disorder_floor)
            & out['entropy_contraction'].ge(params.entropy_drop)
            & out['pattern_7_frequency'].gt(out['pattern_0_frequency'])
            & close.gt(out['close_lag1'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # The prior close's entry instruction fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (np.isfinite(closes[i])
                              and closes[i] >= entry_price * (1.0 + params.profit_target))
                expired = i - entry_bar + 1 >= params.time_stop
                if invalid or target_hit or expired:
                    active = False
                    signals[i] = 0
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=20, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 105.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[5], 'close'] = 104.0
    indicators = pd.DataFrame({'entry': False}, index=index)
    indicators.loc[index[[2, 9]], 'entry'] = True
    params = RecoveryParams(time_stop=3)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    expected = np.zeros(20, dtype=int)
    expected[3:6] = 1
    expected[10:13] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
