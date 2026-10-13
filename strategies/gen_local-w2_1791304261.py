from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    trend_window: int = 12
    noise_window: int = 20
    breakout_window: int = 8
    innovation_threshold: float = 0.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = 'gen_local-w2_1791304261'

    @classmethod
    def params_type(cls) -> type[TrendInnovationParams]:
        return TrendInnovationParams

    def warmup_bars(self, params: TrendInnovationParams) -> int:
        return int(max(params.trend_window, params.noise_window, params.breakout_window) + 2)

    def indicators(self, data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        if min(params.trend_window, params.noise_window, params.breakout_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError('innovation_threshold must be finite and nonnegative.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0))
        out['log_return'] = out['log_close'].diff()
        n = params.trend_window
        position = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        price_sum = out['log_close'].rolling(n, min_periods=n).sum()
        weighted_sum = (position * out['log_close']).rolling(n, min_periods=n).sum()
        center = position - (n - 1) / 2.0
        out['slope'] = (weighted_sum - center * price_sum) / (n * (n * n - 1) / 12.0)
        out['trend_level'] = price_sum / n + out['slope'] * (n - 1) / 2.0
        out['forecast'] = out['trend_level'].shift(1) + out['slope'].shift(1)
        out['innovation'] = out['log_close'] - out['forecast']
        out['return_noise'] = out['log_return'].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1)
        out['innovation_score'] = (
            out['innovation'] / out['return_noise'].where(out['return_noise'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['breakout_level'] = data['high'].rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max().shift(1)
        out['setup'] = (
            (close > out['breakout_level'])
            & (out['slope'] > 0)
            & (out['innovation_score'] > params.innovation_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendInnovationParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        setup = indicators['setup'].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i, ready in enumerate(confirmed):
            if entry_bar >= 0:
                if i - entry_bar < params.hold_bars:
                    signal[i] = 1
                else:
                    # Force an exit bar even when entry conditions persist.
                    entry_bar = -1
            elif ready:
                entry_bar = i
                signal[i] = 1

        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {'setup': [True, True, False, False, False, False, False, True, True, False]},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, TrendInnovationParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    assert (result['size'] > 0).all()
    for hold in (3, 4, 5):
        persistent = pd.DataFrame({'setup': True}, index=index)
        result = strategy.generate_signals(
            data, persistent, None, TrendInnovationParams(hold_bars=hold)
        ).data
        assert result['signal'].iloc[:2].eq(0).all()
        assert result['signal'].iloc[2:2 + hold].eq(1).all()
        assert result['signal'].iloc[2 + hold] == 0


if __name__ == '__main__':
    _self_check()
