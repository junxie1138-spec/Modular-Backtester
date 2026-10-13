from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapExhaustionParams:
    window: int = 5
    exhaustion_ratio: float = 0.65
    pressure_threshold: float = 0.008
    profit_target: float = 0.012
    max_hold: int = 8

    def __post_init__(self):
        if not 3 <= self.window <= 8 or not 5 <= self.max_hold <= 10:
            raise ValueError('window must be 3..8 and max_hold 5..10')
        values = (self.exhaustion_ratio, self.pressure_threshold, self.profit_target)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('thresholds must be finite and positive')
        if self.exhaustion_ratio >= 1:
            raise ValueError('exhaustion_ratio must be below one')


class GeneratedStrategy(BaseStrategy[GapExhaustionParams]):
    strategy_id = 'gen_local_1790961967'

    @classmethod
    def params_type(cls) -> type[GapExhaustionParams]:
        return GapExhaustionParams

    def warmup_bars(self, params: GapExhaustionParams) -> int:
        return int(params.window + 2)

    def indicators(self, data: pd.DataFrame, params: GapExhaustionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['gap'] = (data['open'] / data['close'].shift(1).replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        out['intraday'] = (data['close'] / data['open'].replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        out['prior_gap_scale'] = out['gap'].abs().rolling(params.window, min_periods=params.window).mean().shift(1)
        out['reproduction'] = (out['gap'].abs() / out['prior_gap_scale'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['up_count'] = out['gap'].gt(0).astype(float).where(out['gap'].notna()).rolling(params.window, min_periods=params.window).sum()
        out['down_count'] = out['gap'].lt(0).astype(float).where(out['gap'].notna()).rolling(params.window, min_periods=params.window).sum()
        out['pressure'] = out['gap'].rolling(params.window, min_periods=params.window).sum()
        valid = out.notna().all(axis=1)
        exhausted = valid & out['gap'].abs().ge(0.001) & out['reproduction'].le(params.exhaustion_ratio)
        out['stalled'] = (out['intraday'] * np.sign(out['gap']) <= 0.25 * out['gap'].abs()).astype(bool)
        out['saturated_up'] = out['up_count'].ge(np.ceil(0.8 * params.window)) & out['pressure'].ge(params.pressure_threshold) & out['gap'].gt(0)
        out['saturated_down'] = out['down_count'].ge(np.ceil(0.8 * params.window)) & out['pressure'].le(-params.pressure_threshold) & out['gap'].lt(0)
        out['exhausted'] = exhausted.astype(bool)
        out['entry_direction'] = np.select(
            [out['exhausted'] & out['stalled'] & out['saturated_down'],
             out['exhausted'] & out['stalled'] & out['saturated_up']],
            [1, -1], default=0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapExhaustionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators['entry_direction'].to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position, held = 0, 0
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this bar's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                gain = position * (closes[i] / entry_price - 1) if np.isfinite(entry_price) and entry_price > 0 else np.nan
                # Targets are evaluated at close; exits fill at the next open.
                if not np.isfinite(gain) or gain >= params.profit_target or held >= params.max_hold:
                    position, held = 0, 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                held = 0
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapExhaustionParams(max_hold=5)
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({
        'open': 100.0, 'high': 103.0, 'low': 97.0,
        'close': [100, 100, 102, 100, 100, 98, 100, 100, 100, 100, 100, 100],
        'volume': 1000.0,
    }, index=index)
    entries = pd.DataFrame({'entry_direction': [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    data['close'] = 100.0
    entries['entry_direction'] = [1] + [0] * 11
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) <= 10
    assert strategy.indicators(data, params)['entry_direction'].eq(0).all()


if __name__ == '__main__':
    _self_check()
