from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DeformationParams:
    peak_window: int = 126
    memory_window: int = 18
    percentile_window: int = 252
    depth_percentile: float = 0.80
    repair_percentile: float = 0.70


class GeneratedStrategy(BaseStrategy[DeformationParams]):
    strategy_id = 'gen_local-w2_1791313391'

    @classmethod
    def params_type(cls):
        return DeformationParams

    def warmup_bars(self, params: DeformationParams) -> int:
        return int(params.peak_window + params.memory_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: DeformationParams) -> pd.DataFrame:
        if min(params.peak_window, params.memory_window, params.percentile_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.depth_percentile < 1.0:
            raise ValueError('depth_percentile must be between zero and one.')
        if not 0.5 < params.repair_percentile < 1.0:
            raise ValueError('repair_percentile must be between 0.5 and one.')
        out = pd.DataFrame(index=data.index)
        out['clean_close'] = data['close'].replace([np.inf, -np.inf], np.nan).where(data['close'] > 0)
        out['peak'] = out['clean_close'].rolling(params.peak_window).max()
        out['depth'] = 1.0 - out['clean_close'] / out['peak']
        out['deformation_debt'] = out['depth'].rolling(params.memory_window).mean()
        out['recent_depth'] = out['depth'].rolling(max(1, params.memory_window // 2)).mean()
        out['repair_balance'] = out['deformation_debt'] - out['recent_depth']
        out['depth_threshold'] = out['deformation_debt'].shift(1).rolling(params.percentile_window).quantile(params.depth_percentile)
        history = out['repair_balance'].shift(1).rolling(params.percentile_window)
        out['repair_upper'] = history.quantile(params.repair_percentile)
        out['repair_lower'] = history.quantile(1.0 - params.repair_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DeformationParams,
    ) -> SignalFrame:
        required = ['deformation_debt', 'repair_balance', 'depth_threshold', 'repair_upper', 'repair_lower']
        valid = indicators[required].notna().all(axis=1)
        stressed = indicators['deformation_debt'] > indicators['depth_threshold']
        long_entry = valid & stressed & (indicators['repair_balance'] > indicators['repair_upper'])
        short_entry = valid & stressed & (indicators['repair_balance'] < indicators['repair_lower'])
        df = pd.DataFrame(index=data.index)
        # Each position exits to zero when its entry condition flips false.
        # Monthly depth smoothing sets the target horizon; there is no timed exit.
        df['signal'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = DeformationParams()
    strategy = GeneratedStrategy()
    t = np.arange(900, dtype=float)
    close = 100.0 * np.exp(0.0002 * t + 0.10 * np.sin(t / 45.0) + 0.02 * np.sin(t / 7.0))
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0}, index=pd.date_range('2000-01-01', periods=len(t), freq='B'))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    stressed = indicators['deformation_debt'] > indicators['depth_threshold']
    expected = pd.Series(np.select([stressed & (indicators['repair_balance'] > indicators['repair_upper']), stressed & (indicators['repair_balance'] < indicators['repair_lower'])], [1, -1], default=0), index=data.index).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert set(signals['signal'].unique()) == {-1, 0, 1}
    assert (signals['size'] > 0).all()
    prefix = data.iloc[:650]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:650], prefix_signals)


if __name__ == '__main__':
    _self_check()
