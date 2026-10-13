from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 45
    hold_bars: int = 18

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 5:
            raise ValueError('lookback must be an integer >= 5')
        if not isinstance(self.hold_bars, int) or not 15 <= self.hold_bars <= 20:
            raise ValueError('hold_bars must be an integer between 15 and 20')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791258129'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = data['open'] / out['previous_close'] - 1.0
        out['intraday'] = data['close'] / data['open'].where(data['open'] > 0) - 1.0
        out[['gap', 'intraday']] = out[['gap', 'intraday']].replace([np.inf, -np.inf], np.nan)
        window = out['gap'].rolling(params.lookback, min_periods=params.lookback)
        out['gap_variance'] = window.var(ddof=0)
        out['gap_covariance'] = window.cov(out['intraday'], ddof=0)
        out['response'] = (
            out['gap_covariance'] / out['gap_variance'].where(out['gap_variance'] > 1e-12)
        ).shift(1)
        out['gap_scale'] = np.sqrt(out['gap_variance']).shift(1)
        out['gap_z'] = out['gap'] / out['gap_scale'].where(out['gap_scale'] > 1e-6)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators[['response', 'gap_z', 'intraday']]).all(axis=1)
        continuation = (indicators['response'] >= 0.25) & (indicators['gap_z'] >= 1.0)
        rejection = (indicators['response'] <= -0.25) & (indicators['gap_z'] <= -1.0)
        entries = (
            valid & (continuation | rejection) & (indicators['intraday'] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)

        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                raw[i] = 1
            elif enter:
                raw[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=42, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {'response': 0.5, 'gap_z': 2.0, 'intraday': 0.01}, index=index
    )
    indicators.iloc[2:19, indicators.columns.get_loc('intraday')] = -0.01
    signal = strategy.generate_signals(data, indicators, None, params).data['signal']
    assert signal.iloc[0] == 0
    assert signal.iloc[1:19].eq(1).all()
    assert signal.iloc[19] == 0
    assert signal.iloc[20:38].eq(1).all()
    assert signal.iloc[38] == 0
    indicators['response'] = -0.5
    indicators['gap_z'] = -2.0
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].equals(signal)
    indicators['response'] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()
