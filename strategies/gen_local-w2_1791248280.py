from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalShockParams:
    ordinal_window: int = 24
    volatility_window: int = 20
    complexity_threshold: float = 0.12
    impulse_threshold: float = 0.9
    holding_bars: int = 4

    def __post_init__(self):
        if self.ordinal_window < 3 or self.volatility_window < 2:
            raise ValueError('Lookback windows are too short.')
        if self.holding_bars not in (3, 4, 5):
            raise ValueError('holding_bars must be 3, 4, or 5.')
        if not 0.0 < self.complexity_threshold < 1.0:
            raise ValueError('complexity_threshold must be between zero and one.')
        if not np.isfinite(self.impulse_threshold) or self.impulse_threshold <= 0:
            raise ValueError('impulse_threshold must be finite and positive.')


class GeneratedStrategy(BaseStrategy[OrdinalShockParams]):
    strategy_id = 'gen_local-w2_1791248280'

    @classmethod
    def params_type(cls) -> type[OrdinalShockParams]:
        return OrdinalShockParams

    def warmup_bars(self, params: OrdinalShockParams) -> int:
        return int(max(200, params.ordinal_window + 3, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalShockParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag2'] = out['return'].shift(2)
        out['return_lag1'] = out['return'].shift(1)
        a = out['return_lag2']
        b = out['return_lag1']
        c = out['return']
        valid = a.notna() & b.notna() & c.notna()
        # Ties rank earlier observations first, giving deterministic ordinal motifs.
        out['ordinal_code'] = (
            3 * ((a > b).astype(int) + (a > c).astype(int))
            + (b >= a).astype(int) + (b > c).astype(int)
        ).where(valid)
        probability_columns = []
        for code in (1, 2, 3, 5, 6, 7):
            column = f'probability_{code}'
            probability_columns.append(column)
            occurrence = out['ordinal_code'].eq(code).astype(float).where(valid)
            out[column] = occurrence.rolling(
                params.ordinal_window, min_periods=params.ordinal_window
            ).mean()
        probabilities = out[probability_columns]
        out['entropy'] = -(
            probabilities * np.log(probabilities.clip(lower=1e-12))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out['disequilibrium'] = (
            6.0 * probabilities.pow(2).sum(axis=1, min_count=6) - 1.0
        ) / 5.0
        out['complexity'] = (
            out['entropy'] * out['disequilibrium']
        ).clip(lower=0.0, upper=1.0)
        out['volatility'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out['impulse'] = (
            out['return'] / out['volatility'].where(out['volatility'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalShockParams,
    ) -> SignalFrame:
        ready = indicators[['complexity', 'impulse', 'ma200']].notna().all(axis=1)
        close = data['close']
        regime = np.where(
            close > indicators['ma200'], 1,
            np.where(close < indicators['ma200'], -1, 0),
        )
        impulse_direction = np.sign(indicators['impulse'].fillna(0.0))
        direction = np.where(
            indicators['complexity'] >= params.complexity_threshold,
            -impulse_direction,
            impulse_direction,
        )
        eligible = (
            ready
            & (indicators['impulse'].abs() >= params.impulse_threshold)
            & (direction == regime)
        )
        entries = np.where(eligible, direction, 0).astype(int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0 and i == exit_bar:
                # A flat exit bar prevents immediate reentry from erasing the exit.
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = OrdinalShockParams()
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    data = pd.DataFrame({'close': 101.0}, index=index)
    indicators = pd.DataFrame({
        'ma200': [100.0, 102.0, 102.0, 102.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'complexity': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        'impulse': [2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    data['close'] = 99.0
    indicators['ma200'] = 100.0
    indicators['impulse'] = -indicators['impulse']
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
