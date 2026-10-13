from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 21
    max_complexity: float = 0.65
    profit_target: float = 0.035
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791203492'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.entropy_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.entropy_window < 3 or not 0 < params.max_complexity <= 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        volume = data['volume'].where(np.isfinite(data['volume']) & data['volume'].gt(0))
        a, b, c = volume.shift(2), volume.shift(1), volume
        valid = a.notna() & b.notna() & c.notna()
        # Equal volumes are ordered chronologically for deterministic ties.
        patterns = np.select(
            [a.le(b) & b.le(c), a.le(c) & c.lt(b),
             b.lt(a) & a.le(c), b.le(c) & c.lt(a),
             c.lt(a) & a.le(b)],
            [0, 1, 2, 3, 4], default=5,
        )
        out['volume_pattern'] = pd.Series(patterns, index=data.index).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for pattern in range(6):
            p = out['volume_pattern'].eq(pattern).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            out[f'pattern_probability_{pattern}'] = p
            entropy -= p * np.log(p.where(p.gt(0), 1.0))
        out['complexity'] = entropy / np.log(6.0)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['late_month'] = data.index.day >= 25
        out['mid_month'] = (data.index.day >= 8) & (data.index.day <= 12)
        eligible = out['complexity'].le(params.max_complexity) & out['ma200'].notna()
        out['candidate'] = np.select(
            [eligible & out['late_month'] & data['close'].gt(out['ma200']),
             eligible & out['mid_month'] & data['close'].lt(out['ma200'])],
            [1, -1], default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price = np.nan
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    entry_price, entry_bar = opens[i], i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                # Targets are observed at the close; exits fill next bar.
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position, entry_price = 0, np.nan
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    dates = pd.bdate_range('2020-01-01', periods=240)
    close = np.linspace(90.0, 110.0, 240)
    data = pd.DataFrame(
        {'open': close, 'high': close + 1, 'low': close - 1,
         'close': close, 'volume': np.arange(240) + 1000.0}, index=dates,
    )
    strategy, params = GeneratedStrategy(), Params()
    derived = strategy.indicators(data, params)
    assert derived['complexity'].iloc[:params.entropy_window + 1].isna().all()
    assert np.isclose(derived['complexity'].iloc[-1], 0.0)
    assert derived['candidate'].iloc[:199].eq(0).all()
    pd.testing.assert_frame_equal(
        derived.iloc[:220], strategy.indicators(data.iloc[:220], params)
    )
    test = data.iloc[:13].copy()
    test.loc[:, 'open'] = 100.0
    test.loc[:, 'close'] = 100.0
    test.iloc[2, test.columns.get_loc('close')] = 104.0
    test.iloc[6, test.columns.get_loc('close')] = 96.0
    candidates = pd.DataFrame(
        {'candidate': [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0]},
        index=test.index,
    )
    result = strategy.generate_signals(
        test, candidates, None, Params(max_hold_bars=3)
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
