from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 30
    impact_window: int = 3
    entry_z: float = 1.2
    impact_ratio_max: float = 0.55
    hold_bars: int = 8

    def __post_init__(self):
        if self.ma_window < 2 or self.impact_window < 1 or self.hold_bars < 1:
            raise ValueError('Invalid lookback or holding period')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be positive and finite')
        if not np.isfinite(self.impact_ratio_max) or not 0 < self.impact_ratio_max < 1:
            raise ValueError('impact_ratio_max must lie between zero and one')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791318294'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window, 2 * params.impact_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] > 0)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z_score'] = (close - out['ma']) / out['price_std'].where(out['price_std'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['dollar_volume'] = close * volume
        out['downside_impact'] = (-out['return']).clip(lower=0) / out['dollar_volume']
        out['recent_impact'] = out['downside_impact'].rolling(
            params.impact_window, min_periods=params.impact_window
        ).mean()
        out['prior_impact'] = out['recent_impact'].shift(params.impact_window)
        out['impact_ratio'] = out['recent_impact'] / out['prior_impact'].where(out['prior_impact'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry = (
            (indicators['z_score'] <= -params.entry_z)
            & (indicators['recent_impact'] > 0)
            & (indicators['prior_impact'] > 0)
            & (indicators['impact_ratio'] <= params.impact_ratio_max)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, eligible in enumerate(entry):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat bar even when entry conditions still agree.
                    exit_bar = -1
            elif eligible:
                exit_bar = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    for hold in (6, 8, 10):
        params = StrategyParams(hold_bars=hold)
        index = pd.date_range('2020-01-01', periods=3 * (hold + 1), freq='B')
        data = pd.DataFrame(
            {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0},
            index=index,
        )
        ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
        indicators = pd.DataFrame(
            {'z_score': -2.0, 'recent_impact': 0.4, 'prior_impact': 1.0, 'impact_ratio': 0.4},
            index=index,
        )
        signals = strategy.generate_signals(data, indicators, ctx, params).data
        assert signals['signal'].tolist() == ([0] + [1] * hold) * 3
        assert signals['size'].gt(0).all()
        actual = strategy.indicators(data, params)
        assert strategy.generate_signals(data, actual, ctx, params).data['signal'].eq(0).all()
