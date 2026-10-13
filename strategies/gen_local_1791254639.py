from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 126
    volume_window: int = 5
    min_depth: float = 0.03
    regime_depth: float = 0.08
    max_depth: float = 0.18
    balanced_share: float = 0.30
    burst_share: float = 0.45
    holding_bars: int = 8

    def __post_init__(self):
        if self.peak_window < 2 or self.volume_window < 2 or self.holding_bars < 1:
            raise ValueError('Windows must exceed one and holding_bars must be positive.')
        if not 0 < self.min_depth < self.regime_depth < self.max_depth < 1:
            raise ValueError('Drawdown thresholds must be ordered between zero and one.')
        if not 0 < self.balanced_share < self.burst_share <= 1:
            raise ValueError('Turnover thresholds must be ordered between zero and one.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791254639'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.peak_window, params.volume_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['peak'] = data['close'].where(data['close'] > 0).rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        volume = data['volume'].where(data['volume'] >= 0)
        out['volume_sum'] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).sum()
        out['volume_max'] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).max()
        out['depth'] = (1.0 - data['close'] / out['peak']).clip(lower=0)
        out['turnover_share'] = out['volume_max'] / out['volume_sum'].where(out['volume_sum'] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['moderate_regime'] = (
            out['depth'].ge(params.min_depth)
            & out['depth'].lt(params.regime_depth)
        )
        out['deep_regime'] = (
            out['depth'].ge(params.regime_depth)
            & out['depth'].le(params.max_depth)
        )
        out['entry'] = (
            (out['moderate_regime'] & out['turnover_share'].le(params.balanced_share))
            | (out['deep_regime'] & out['turnover_share'].ge(params.burst_share))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, eligible in enumerate(entry):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif eligible:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame(
        {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 100.0},
        index=index,
    )
    entries = pd.DataFrame({'entry': True}, index=index)
    signal = strategy.generate_signals(data, entries, None, params).data['signal']
    assert signal.iloc[0] == 0
    assert signal.iloc[1:9].eq(1).all()
    assert signal.iloc[9] == 0
    assert signal.iloc[10:18].eq(1).all()
    assert signal.iloc[18] == 0
    entries['entry'] = False
    entries.iloc[0, 0] = True
    signal = strategy.generate_signals(data, entries, None, params).data['signal']
    assert signal.iloc[1:9].eq(1).all()
    assert signal.iloc[9:].eq(0).all()
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
