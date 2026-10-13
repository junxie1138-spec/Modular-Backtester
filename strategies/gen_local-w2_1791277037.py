from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    drawdown_window: int = 20
    gap_window: int = 10
    min_drawdown: float = 0.006
    min_gap_noise: float = 0.25
    recovery_fraction: float = 0.6
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = 'gen_local-w2_1791277037'

    @classmethod
    def params_type(cls) -> type[GapRecoveryParams]:
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return int(max(params.drawdown_window, params.gap_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        if params.drawdown_window < 2 or params.gap_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        numeric = np.array([params.min_drawdown, params.min_gap_noise,
                            params.recovery_fraction, params.max_size], dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError('Parameters must be finite.')
        if params.min_drawdown <= 0 or params.min_gap_noise <= 0:
            raise ValueError('Drawdown and gap thresholds must be positive.')
        if params.recovery_fraction <= 0 or not 0 < params.max_size <= 1:
            raise ValueError('Recovery fraction must be positive and size within (0, 1].')

        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1).where(lambda x: x > 0)
        out['prior_peak'] = data['close'].rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max().shift(1)
        out['drawdown'] = 1.0 - out['prior_close'] / out['prior_peak'].replace(0, np.nan)
        out['gap'] = (data['open'] - out['prior_close']) / out['prior_close']
        out['gap_noise'] = out['gap'].abs().rolling(
            params.gap_window, min_periods=params.gap_window
        ).mean().shift(1).replace(0, np.nan)
        out['session_move'] = (data['close'] - data['open']) / out['prior_close']
        out['repair_fraction'] = out['session_move'].abs() / out['gap'].abs().replace(0, np.nan)
        out['strength'] = (out['session_move'].abs() / (2.0 * out['gap_noise'])).clip(0.2, 1.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        eligible = (
            (out['drawdown'] >= params.min_drawdown)
            & (out['gap'].abs() >= params.min_gap_noise * out['gap_noise'])
            & (out['repair_fraction'] >= params.recovery_fraction)
            & out['strength'].notna()
        )
        bullish = eligible & (out['gap'] < 0) & (out['session_move'] > 0)
        bearish = eligible & (out['gap'] > 0) & (out['session_move'] < 0)
        out['evidence'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRecoveryParams,
    ) -> SignalFrame:
        evidence = indicators['evidence'].fillna(0).to_numpy(dtype=int)
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.2).clip(0.2, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size * 0.2, dtype=float)
        position = 0
        entry_size = params.max_size * 0.2
        for i, direction in enumerate(evidence):
            if position != 0:
                if direction == -position:
                    position = 0
            elif direction != 0:
                position = int(direction)
                entry_size = float(params.max_size * strength[i])
            signals[i] = position
            sizes[i] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.max_size * 0.2).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'evidence': [1, 0, -1, -1, 0, 1, 1],
        'strength': [0.4, 0.9, 0.8, 0.7, 0.3, 0.9, 0.5],
    }, index=index)
    params = GapRecoveryParams()
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result['size'].iloc[[1, 2]], params.max_size * 0.4)
    assert np.allclose(result['size'].iloc[[4, 5]], params.max_size * 0.7)
    assert np.isfinite(result['size']).all() and (result['size'] > 0).all()
