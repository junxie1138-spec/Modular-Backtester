from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_streak: int = 3
    contraction: float = 0.6
    recovery_body: float = 0.25
    profit_target: float = 0.012
    time_stop: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1790907111'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not 2 <= params.min_streak <= 6:
            raise ValueError('min_streak must be between 2 and 6')
        if not 3 <= params.time_stop <= 5:
            raise ValueError('time_stop must be between 3 and 5')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if not 0 < params.contraction <= 1 or not 0 < params.recovery_body <= 1:
            raise ValueError('contraction and recovery_body must be in (0, 1]')

        out = pd.DataFrame(index=data.index)
        out['move'] = data['close'].diff()
        out['previous_move'] = out['move'].shift(1)
        down = out['move'].lt(0)
        up = out['move'].gt(0)
        down_run = pd.Series(True, index=data.index)
        up_run = pd.Series(True, index=data.index)
        out['down_streak'] = 0
        out['up_streak'] = 0
        for lag in range(6):
            down_run &= down.shift(lag, fill_value=False)
            up_run &= up.shift(lag, fill_value=False)
            out['down_streak'] += down_run.astype(int)
            out['up_streak'] += up_run.astype(int)

        out['range'] = (data['high'] - data['low']).where(lambda x: x > 0)
        out['recovery_body'] = ((data['close'] - data['open']) / out['range']).replace([np.inf, -np.inf], np.nan)
        out['contracting'] = (
            out['previous_move'].abs().gt(0)
            & out['move'].abs().le(params.contraction * out['previous_move'].abs())
        )
        out['valid'] = (
            np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
            & data['open'].gt(0)
            & data['close'].gt(0)
            & out['range'].notna()
        )
        long_entry = (
            out['valid'] & out['contracting']
            & out['down_streak'].ge(params.min_streak)
            & out['recovery_body'].ge(params.recovery_body)
        )
        short_entry = (
            out['valid'] & out['contracting']
            & out['up_streak'].ge(params.min_streak)
            & out['recovery_body'].le(-params.recovery_body)
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        held = 0

        for i in range(len(data)):
            if not np.isfinite(opens[i]) or not np.isfinite(closes[i]) or opens[i] <= 0 or closes[i] <= 0:
                position = 0
                entry_price = np.nan
                held = 0
                continue
            if position:
                # The previous close's decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # Targets are evaluated at the close; exits fill next open.
                if gain >= params.profit_target or held >= params.time_stop:
                    position = 0
                    entry_price = np.nan
                    held = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
                held = 0
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=12)
    close = pd.Series([105, 103, 101, 100, 100, 100, 100, 100, 100, 100, 100, 100], index=index, dtype=float)
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
    data.loc[index[3], ['open', 'low', 'high']] = [99.0, 98.5, 100.5]
    derived = strategy.indicators(data, params)
    assert derived.loc[index[3], 'down_streak'] == 3
    assert derived.loc[index[3], 'entry'] == 1
    assert strategy.warmup_bars(params) <= 10

    entries = pd.DataFrame({'entry': [1] + [0] * 11}, index=index)
    flat = data.copy()
    flat[['open', 'close']] = 100.0
    result = strategy.generate_signals(flat, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    for direction in (1, -1):
        entries['entry'] = [direction] + [0] * 11
        target_data = flat.copy()
        target_data.loc[index[1], 'close'] = 100.0 * (1 + direction * 0.02)
        result = strategy.generate_signals(target_data, entries, None, params).data
        assert result['signal'].tolist() == [0, direction] + [0] * 10
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
