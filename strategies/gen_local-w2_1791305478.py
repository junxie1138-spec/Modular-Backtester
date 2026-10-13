from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInversionParams:
    percentile_window: int = 63
    gap_quantile: float = 0.85
    flow_window: int = 5
    vol_window: int = 20
    target_vol: float = 0.12
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapInversionParams]):
    strategy_id = 'gen_local-w2_1791305478'

    @classmethod
    def params_type(cls) -> type[GapInversionParams]:
        return GapInversionParams

    def warmup_bars(self, params: GapInversionParams) -> int:
        return int(max(params.percentile_window + 2, params.flow_window + 2, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapInversionParams) -> pd.DataFrame:
        if min(params.percentile_window, params.flow_window, params.vol_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.gap_quantile < 1.0:
            raise ValueError('gap_quantile must be between zero and one.')
        if not np.isfinite(params.target_vol) or params.target_vol <= 0.0:
            raise ValueError('target_vol must be finite and positive.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda x: x > 0.0)
        opening = data['open'].where(lambda x: x > 0.0)
        out['gap'] = opening.div(previous_close).sub(1.0)
        out['gap_threshold'] = out['gap'].abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.gap_quantile).shift(1)
        out['overnight_flow'] = out['gap'].rolling(params.flow_window, min_periods=params.flow_window).sum().shift(1)
        out['intraday_return'] = data['close'].div(opening).sub(1.0)
        out['close_return'] = data['close'].div(previous_close).sub(1.0)
        out['annual_vol'] = out['close_return'].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0).mul(np.sqrt(252.0))
        out['target_size'] = params.target_vol / out['annual_vol'].where(out['annual_vol'] > 0.0)
        out['target_size'] = out['target_size'].clip(lower=0.05, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapInversionParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        columns = ['gap', 'gap_threshold', 'overnight_flow', 'intraday_return', 'target_size']
        ready = pd.Series(np.isfinite(indicators[columns].to_numpy(dtype=float)).all(axis=1), index=data.index)
        gap = indicators['gap']
        eligible = ready & gap.abs().gt(indicators['gap_threshold']) & gap.mul(indicators['overnight_flow']).lt(0.0) & gap.mul(indicators['intraday_return']).gt(0.0)
        entries = pd.Series(0, index=data.index, dtype=int)
        entries.loc[eligible & gap.gt(0.0)] = 1
        entries.loc[eligible & gap.lt(0.0)] = -1
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        entry_values = entries.to_numpy()
        size_values = indicators['target_size'].to_numpy(dtype=float)
        position = 0
        exit_at = -1
        held_size = 0.95
        for i in range(len(data)):
            if position:
                if i >= exit_at:
                    position = 0
            elif entry_values[i] != 0:
                position = int(entry_values[i])
                exit_at = i + params.hold_bars
                held_size = float(np.clip(size_values[i], 0.05, 0.95))
            signals[i] = position
            if position:
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    gaps = np.array([0.02, -0.02, -0.02, -0.02, 0.02, -0.02, 0.02])
    indicators = pd.DataFrame({'gap': gaps, 'gap_threshold': 0.01, 'overnight_flow': -gaps, 'intraday_return': gaps, 'target_size': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3]}, index=index)
    strategy = GeneratedStrategy()
    for hold, expected in [(2, [0, 1, 1, 0, -1, -1, 0]), (1, [0, 1, 0, -1, 0, 1, 0])]:
        result = strategy.generate_signals(data, indicators, None, GapInversionParams(hold_bars=hold)).data
        assert result['signal'].tolist() == expected
        assert np.isfinite(result['size']).all() and result['size'].gt(0.0).all()
        if hold == 2:
            assert np.allclose(result['size'], [0.95, 0.4, 0.4, 0.95, 0.7, 0.7, 0.95])
    indicators.loc[:, 'gap_threshold'] = np.nan
    assert strategy.generate_signals(data, indicators, None, GapInversionParams()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
